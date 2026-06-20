from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import json
from collections import defaultdict
import base64
from io import BytesIO
from PIL import Image
import uvicorn
import os
import google.generativeai as genai

app = FastAPI(title="School Dining Waste Tracker API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("Warning: GEMINI_API_KEY not set. Using fallback detection.")
    gemini_model = None

# In-memory storage (replace with database for production)
scans_db = []
daily_reports_db = {}

# Constants
WASTE_LEVELS = {
    0.0: "None",
    0.1: "Minimal",
    0.25: "Moderate",
    0.40: "Significant",
    1.0: "Most Left"
}


def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


async def analyze_plate_with_gemini(before_img: np.ndarray, after_img: np.ndarray) -> Dict:
    """
    Use Gemini Vision API to analyze the plate before and after eating.
    Returns detailed analysis of each food item and waste estimation.
    """
    if not gemini_model:
        return use_fallback_detection(before_img, after_img)
    
    try:
        # Convert images to PIL format for Gemini
        before_pil = cv2_to_pil(before_img)
        after_pil = cv2_to_pil(after_img)
        
        # Create detailed prompt for Gemini
        prompt = """Analyze these two images of a dining plate - the first shows the plate before eating, 
        the second shows the same plate after eating.
        
        Please provide a detailed JSON response with the following structure:
        {
            "food_items": [
                {
                    "name": "food item name",
                    "initial_portion": "description of initial amount (e.g., 'full serving', '6 oz')",
                    "remaining_portion": "description of remaining amount",
                    "waste_percentage": <number between 0-100 representing how much was LEFT/WASTED, not eaten>,
                    "estimated_weight_oz": <estimated weight of WASTED food in ounces>,
                    "category": "entree/side/vegetable/dessert/beverage"
                }
            ],
            "overall_assessment": "brief summary of waste patterns",
            "suggestions": ["actionable tip 1", "actionable tip 2"]
        }
        
        IMPORTANT: waste_percentage should represent the percentage of food that was LEFT ON THE PLATE (wasted), 
        not the percentage that was eaten. For example:
        - If someone ate everything: waste_percentage = 0-10%
        - If someone ate most of it: waste_percentage = 10-25%
        - If someone ate half: waste_percentage = 40-60%
        - If someone barely touched it: waste_percentage = 75-100%
        
        Be specific about each distinct food item you can identify. Focus on accuracy and be realistic 
        about portion sizes typical in college dining halls."""
        
        # Generate analysis
        response = gemini_model.generate_content([prompt, before_pil, after_pil])
        
        # Parse JSON response
        response_text = response.text
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        analysis = json.loads(response_text)
        return analysis
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
        return use_fallback_detection(before_img, after_img)


def use_fallback_detection(before_img: np.ndarray, after_img: np.ndarray) -> Dict:
    """
    Fallback detection using traditional CV when Gemini is unavailable.
    """
    waste_pct = estimate_waste_percentage_cv(before_img, after_img)
    
    return {
        "food_items": [
            {
                "name": "Mixed Plate",
                "initial_portion": "Full serving",
                "remaining_portion": f"{int((1 - waste_pct) * 100)}% remaining",
                "waste_percentage": round((1 - waste_pct) * 100, 1),
                "estimated_weight_oz": round(8 * (1 - waste_pct), 2),
                "category": "mixed"
            }
        ],
        "overall_assessment": f"Approximately {int((waste_pct) * 100)}% of food was consumed, {int((1- waste_pct) * 100)}% wasted.",
        "suggestions": generate_tips_from_waste(waste_pct)
    }


def estimate_waste_percentage_cv(before_img: np.ndarray, after_img: np.ndarray) -> float:
    """
    Estimate waste percentage using computer vision (fallback method).
    Compares the amount of food before vs after eating.
    """
    try:
        # Resize images to same size if needed
        h, w = before_img.shape[:2]
        after_img = cv2.resize(after_img, (w, h))
        
        # Convert to LAB color space for better food detection
        before_lab = cv2.cvtColor(before_img, cv2.COLOR_BGR2LAB)
        after_lab = cv2.cvtColor(after_img, cv2.COLOR_BGR2LAB)
        
        # Get L channel (lightness) - plates are usually lighter than food
        before_l = before_lab[:, :, 0]
        after_l = after_lab[:, :, 0]
        
        # Get A and B channels (color) - food has more color than white plates
        before_a = before_lab[:, :, 1]
        before_b = before_lab[:, :, 2]
        after_a = after_lab[:, :, 1]
        after_b = after_lab[:, :, 2]
        
        # Calculate color intensity (food has more color variation)
        before_color = np.abs(before_a - 128) + np.abs(before_b - 128)
        after_color = np.abs(after_a - 128) + np.abs(after_b - 128)
        
        # Threshold to detect food (areas with significant color)
        before_mask = before_color > 15  # Areas with food have more color
        after_mask = after_color > 15
        
        # Calculate food area
        before_food_pixels = np.sum(before_mask)
        after_food_pixels = np.sum(after_mask)
        
        print(f"DEBUG CV: Before pixels: {before_food_pixels}, After pixels: {after_food_pixels}")
        
        if before_food_pixels == 0:
            return 0.0
        
        # Waste percentage = food remaining after eating / original food amount
        waste_percentage = after_food_pixels / before_food_pixels
        waste_percentage = max(0.0, min(1.0, waste_percentage))
        
        print(f"DEBUG CV: Calculated waste: {waste_percentage * 100:.1f}%")
        
        return waste_percentage
    except Exception as e:
        print(f"Error in waste estimation: {e}")
        import traceback
        traceback.print_exc()
        return 0.5


def classify_waste_level(waste_percentage: float) -> str:
    """Classify waste percentage into predefined levels."""
    for threshold in sorted(WASTE_LEVELS.keys()):
        if waste_percentage <= threshold:
            return WASTE_LEVELS[threshold]
    return "Most Left"


def calculate_impact(food_items: List[Dict]) -> Dict:
    """Calculate environmental and financial impact of waste."""
    total_weight_oz = sum(item.get("estimated_weight_oz", 0) for item in food_items)
    total_weight_lbs = total_weight_oz / 16
    
    # Average cost per lb of prepared food
    cost_per_lb = 5.50
    
    # CO2 emissions: ~2 kg per lb of food waste
    co2_kg = total_weight_lbs * 2
    
    # Water usage: ~25 gallons per lb of food produced
    water_gallons = total_weight_lbs * 25
    
    return {
        "weight_lbs": round(total_weight_lbs, 3),
        "weight_oz": round(total_weight_oz, 2),
        "cost_usd": round(total_weight_lbs * cost_per_lb, 2),
        "co2_kg": round(co2_kg, 2),
        "water_gallons": round(water_gallons, 1),
        "meals_equivalent": round(total_weight_lbs / 0.75, 2)  # ~0.75 lbs per meal
    }


def calculate_points(food_items: List[Dict]) -> int:
    """Calculate gamification points based on waste across all items."""
    if not food_items:
        return 0
    
    avg_waste = sum(item.get("waste_percentage", 0) for item in food_items) / len(food_items)
    
    # Points scale
    if avg_waste <= 10:
        return 15
    elif avg_waste <= 25:
        return 10
    elif avg_waste <= 40:
        return 5
    elif avg_waste <= 60:
        return 2
    else:
        return 1


def generate_tips_from_waste(waste_pct: float) -> List[str]:
    """Generate tips based on waste percentage."""
    if waste_pct <= 0.1:
        return ["🎉 Amazing job! Clean plate champion!"]
    elif waste_pct <= 0.25:
        return ["Great effort! Keep it up.", "You're being mindful of portions."]
    elif waste_pct <= 0.40:
        return ["💡 Try taking smaller portions initially.", "You can always go back for seconds!"]
    else:
        return [
            "💡 Consider starting with half portions.",
            "Ask dining staff about smaller serving options.",
            "Try one item at a time - you can always get more!"
        ]


@app.post("/api/scan")
async def process_scan(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    student_id: Optional[str] = None,
    school_id: str = "school_001"
):
    """
    Process tray scan and analyze waste using Gemini Vision API.
    Returns detailed breakdown by food item.
    """
    try:
        # Read images
        before_bytes = await before_image.read()
        after_bytes = await after_image.read()
        
        before_img = cv2.imdecode(np.frombuffer(before_bytes, np.uint8), cv2.IMREAD_COLOR)
        after_img = cv2.imdecode(np.frombuffer(after_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if before_img is None or after_img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Analyze with Gemini
        analysis = await analyze_plate_with_gemini(before_img, after_img)
        
        # Calculate overall metrics
        food_items = analysis.get("food_items", [])
        
        # Calculate average waste across all items
        if food_items:
            total_waste = sum(item.get("waste_percentage", 0) for item in food_items)
            avg_waste_pct = total_waste / len(food_items)
        else:
            avg_waste_pct = 0
        
        waste_level = classify_waste_level(avg_waste_pct / 100)
        
        # Calculate environmental impact
        impact = calculate_impact(food_items)
        
        # Calculate points
        points = calculate_points(food_items)
        
        # Store scan
        scan_record = {
            "id": len(scans_db) + 1,
            "timestamp": datetime.now().isoformat(),
            "school_id": school_id,
            "student_id": student_id,
            "food_items": food_items,
            "avg_waste_percentage": round(avg_waste_pct, 2),
            "waste_level": waste_level,
            "points": points,
            "impact": impact,
            "overall_assessment": analysis.get("overall_assessment", ""),
            "suggestions": analysis.get("suggestions", []),
            "before_image": base64.b64encode(before_bytes).decode(),
            "after_image": base64.b64encode(after_bytes).decode()
        }
        scans_db.append(scan_record)
        
        return JSONResponse({
            "success": True,
            "scan_id": scan_record["id"],
            "food_items": food_items,
            "waste_level": waste_level,
            "avg_waste_percentage": round(avg_waste_pct, 1),
            "points": points,
            "impact": impact,
            "overall_assessment": analysis.get("overall_assessment", ""),
            "tips": analysis.get("suggestions", [])
        })
    
    except Exception as e:
        print(f"Error in process_scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/daily-report")
async def get_daily_report(school_id: str = "school_001", date: Optional[str] = None):
    """
    Get daily waste report with dish-level breakdown.
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Filter scans for the day
    target_date = datetime.strptime(date, "%Y-%m-%d").date()
    daily_scans = [
        s for s in scans_db
        if s["school_id"] == school_id and
        datetime.fromisoformat(s["timestamp"]).date() == target_date
    ]
    
    if not daily_scans:
        return JSONResponse({
            "date": date,
            "school_id": school_id,
            "total_scans": 0,
            "data": None
        })
    
    # Aggregate by food item across all scans
    food_stats = defaultdict(lambda: {
        "count": 0, 
        "total_waste_pct": 0, 
        "total_weight_oz": 0,
        "categories": set()
    })
    
    for scan in daily_scans:
        for item in scan.get("food_items", []):
            name = item.get("name", "Unknown")
            food_stats[name]["count"] += 1
            food_stats[name]["total_waste_pct"] += item.get("waste_percentage", 0)
            food_stats[name]["total_weight_oz"] += item.get("estimated_weight_oz", 0)
            food_stats[name]["categories"].add(item.get("category", "other"))
    
    # Calculate averages and create summary
    food_summary = []
    for food_name, stats in food_stats.items():
        avg_waste = stats["total_waste_pct"] / stats["count"]
        food_summary.append({
            "food": food_name,
            "appearances": stats["count"],
            "avg_waste_pct": round(avg_waste, 1),
            "total_wasted_oz": round(stats["total_weight_oz"], 2),
            "category": list(stats["categories"])[0] if stats["categories"] else "other",
            "recommendation": generate_food_recommendation(food_name, avg_waste)
        })
    
    food_summary.sort(key=lambda x: x["avg_waste_pct"], reverse=True)
    
    # Calculate totals
    total_weight = sum(s["impact"]["weight_lbs"] for s in daily_scans)
    total_cost = sum(s["impact"]["cost_usd"] for s in daily_scans)
    total_co2 = sum(s["impact"]["co2_kg"] for s in daily_scans)
    total_water = sum(s["impact"].get("water_gallons", 0) for s in daily_scans)
    avg_waste = sum(s["avg_waste_percentage"] for s in daily_scans) / len(daily_scans)
    
    return JSONResponse({
        "date": date,
        "school_id": school_id,
        "total_scans": len(daily_scans),
        "avg_waste_pct": round(avg_waste, 1),
        "totals": {
            "weight_lbs": round(total_weight, 2),
            "cost_usd": round(total_cost, 2),
            "co2_kg": round(total_co2, 2),
            "water_gallons": round(total_water, 1)
        },
        "by_food": food_summary[:10]  # Top 10 most wasted
    })


@app.get("/api/student-stats")
async def get_student_stats(student_id: str, days: int = 7):
    """
    Get individual student statistics and progress.
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    student_scans = [
        s for s in scans_db
        if s.get("student_id") == student_id and
        datetime.fromisoformat(s["timestamp"]) > cutoff_date
    ]
    
    if not student_scans:
        return JSONResponse({
            "student_id": student_id,
            "scans": 0,
            "message": "No scans found for this period"
        })
    
    # Calculate stats
    total_points = sum(s["points"] for s in student_scans)
    avg_waste = sum(s["avg_waste_percentage"] for s in student_scans) / len(student_scans)
    total_impact = {
        "weight_lbs": sum(s["impact"]["weight_lbs"] for s in student_scans),
        "cost_usd": sum(s["impact"]["cost_usd"] for s in student_scans),
        "co2_kg": sum(s["impact"]["co2_kg"] for s in student_scans)
    }
    
    # Track most wasted foods
    personal_offenders = defaultdict(lambda: {"count": 0, "waste": 0})
    for scan in student_scans:
        for item in scan.get("food_items", []):
            name = item.get("name")
            personal_offenders[name]["count"] += 1
            personal_offenders[name]["waste"] += item.get("waste_percentage", 0)
    
    most_wasted = [
        {
            "food": name,
            "times_wasted": stats["count"],
            "avg_waste_pct": round(stats["waste"] / stats["count"], 1)
        }
        for name, stats in personal_offenders.items()
        if stats["waste"] / stats["count"] > 30  # Only show items with >30% waste
    ]
    most_wasted.sort(key=lambda x: x["avg_waste_pct"], reverse=True)
    
    return JSONResponse({
        "student_id": student_id,
        "period_days": days,
        "total_scans": len(student_scans),
        "total_points": total_points,
        "avg_waste_pct": round(avg_waste, 1),
        "total_impact": {
            "weight_lbs": round(total_impact["weight_lbs"], 2),
            "cost_saved": round(total_impact["cost_usd"], 2),
            "co2_prevented": round(total_impact["co2_kg"], 2)
        },
        "foods_to_avoid": most_wasted[:5],
        "badge": assign_badge(avg_waste),
        "next_goal": get_next_goal(total_points)
    })


@app.get("/api/weekly-report")
async def get_weekly_report(school_id: str = "school_001", weeks_back: int = 0):
    """
    Get week-over-week trends and recommendations.
    """
    end_date = datetime.now() - timedelta(weeks=weeks_back)
    start_date = end_date - timedelta(days=7)
    
    weekly_scans = [
        s for s in scans_db
        if s["school_id"] == school_id and
        start_date <= datetime.fromisoformat(s["timestamp"]) <= end_date
    ]
    
    if not weekly_scans:
        return JSONResponse({
            "week": start_date.strftime("%Y-%m-%d"),
            "data": None
        })
    
    # Daily breakdown
    daily_breakdown = defaultdict(lambda: {"count": 0, "waste": 0, "cost": 0})
    for scan in weekly_scans:
        date_key = datetime.fromisoformat(scan["timestamp"]).strftime("%Y-%m-%d")
        daily_breakdown[date_key]["count"] += 1
        daily_breakdown[date_key]["waste"] += scan["avg_waste_percentage"]
        daily_breakdown[date_key]["cost"] += scan["impact"]["cost_usd"]
    
    daily_data = []
    for date_key in sorted(daily_breakdown.keys()):
        stats = daily_breakdown[date_key]
        daily_data.append({
            "date": date_key,
            "scans": stats["count"],
            "avg_waste_pct": round(stats["waste"] / stats["count"], 1),
            "cost_usd": round(stats["cost"], 2)
        })
    
    # Top food offenders
    food_performance = defaultdict(lambda: {"count": 0, "total_waste": 0})
    for scan in weekly_scans:
        for item in scan.get("food_items", []):
            food = item.get("name", "Unknown")
            food_performance[food]["count"] += 1
            food_performance[food]["total_waste"] += item.get("waste_percentage", 0)
    
    top_offenders = [
        {
            "food": food,
            "avg_waste_pct": round(stats["total_waste"] / stats["count"], 1),
            "appearances": stats["count"]
        }
        for food, stats in food_performance.items()
    ]
    top_offenders.sort(key=lambda x: x["avg_waste_pct"], reverse=True)
    
    return JSONResponse({
        "week_start": start_date.strftime("%Y-%m-%d"),
        "week_end": end_date.strftime("%Y-%m-%d"),
        "total_scans": len(weekly_scans),
        "daily_breakdown": daily_data,
        "top_offenders": top_offenders[:10],
        "recommendations": generate_weekly_recommendations(top_offenders)
    })


@app.get("/api/insights")
async def get_insights(school_id: str = "school_001", days: int = 30):
    """
    Get AI-generated insights and recommendations for dining staff.
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_scans = [
        s for s in scans_db
        if s["school_id"] == school_id and
        datetime.fromisoformat(s["timestamp"]) > cutoff_date
    ]
    
    if not recent_scans:
        return JSONResponse({"insights": []})
    
    insights = []
    
    # Aggregate food waste across all scans
    food_waste = defaultdict(lambda: {"waste": [], "count": 0})
    for scan in recent_scans:
        for item in scan.get("food_items", []):
            food_name = item.get("name", "Unknown")
            food_waste[food_name]["waste"].append(item.get("waste_percentage", 0))
            food_waste[food_name]["count"] += 1
    
    # Insight 1: Highest waste food
    worst_food = max(food_waste.items(), key=lambda x: np.mean(x[1]["waste"]))
    avg_waste = np.mean(worst_food[1]["waste"])
    if avg_waste > 30 and worst_food[1]["count"] > 5:
        insights.append({
            "type": "alert",
            "title": f"High Waste Alert: {worst_food[0]}",
            "description": f"{worst_food[0]} shows {int(avg_waste)}% average waste across {worst_food[1]['count']} servings. Consider smaller portions or menu substitution.",
            "priority": "high",
            "action": "reduce_portion",
            "data": {
                "food": worst_food[0],
                "avg_waste_pct": round(avg_waste, 1),
                "servings": worst_food[1]["count"]
            }
        })
    
    # Insight 2: Best performing food
    best_food = min(food_waste.items(), key=lambda x: np.mean(x[1]["waste"]))
    if best_food[1]["count"] > 5:
        insights.append({
            "type": "success",
            "title": f"Popular Choice: {best_food[0]}",
            "description": f"{best_food[0]} has only {int(np.mean(best_food[1]['waste']))}% waste. Students love this option!",
            "priority": "medium",
            "data": {
                "food": best_food[0],
                "avg_waste_pct": round(np.mean(best_food[1]["waste"]), 1)
            }
        })
    
    # Insight 3: Monthly impact
    total_weight = sum(s["impact"]["weight_lbs"] for s in recent_scans)
    total_cost = sum(s["impact"]["cost_usd"] for s in recent_scans)
    total_co2 = sum(s["impact"]["co2_kg"] for s in recent_scans)
    
    insights.append({
        "type": "info",
        "title": "Monthly Environmental Impact",
        "description": f"{int(total_weight)} lbs of food wasted, ${int(total_cost)} lost, {int(total_co2)} kg CO2 emitted. Reducing waste by 20% would save ${int(total_cost * 0.2)}/month.",
        "priority": "info",
        "data": {
            "weight_lbs": round(total_weight, 1),
            "cost_usd": round(total_cost, 2),
            "co2_kg": round(total_co2, 1)
        }
    })
    
    # Insight 4: Day of week patterns
    daily_waste = defaultdict(list)
    for scan in recent_scans:
        day = datetime.fromisoformat(scan["timestamp"]).strftime("%A")
        daily_waste[day].append(scan["avg_waste_percentage"])
    
    if daily_waste:
        best_day = min(daily_waste.items(), key=lambda x: np.mean(x[1]))
        insights.append({
            "type": "info",
            "title": f"{best_day[0]} Success",
            "description": f"{best_day[0]} has the lowest waste at {int(np.mean(best_day[1]))}%. Consider analyzing this day's menu for successful patterns.",
            "priority": "low"
        })
    
    return JSONResponse({"insights": insights})


def generate_food_recommendation(food: str, avg_waste: float) -> str:
    """Generate dining staff recommendations for specific food."""
    if avg_waste > 50:
        return f"⚠️ High waste ({int(avg_waste)}%). Consider removing or replacing."
    elif avg_waste > 35:
        return f"⚡ Reduce portion size by 30-40%."
    elif avg_waste > 20:
        return f"📊 Monitor closely. Offer smaller portion option."
    else:
        return f"✓ Popular item ({int(avg_waste)}% waste). Maintain current approach."


def generate_weekly_recommendations(top_offenders: List[dict]) -> List[str]:
    """Generate strategic recommendations for dining staff."""
    recommendations = []
    
    if top_offenders and len(top_offenders) > 0:
        top_food = top_offenders[0]
        if top_food["avg_waste_pct"] > 40:
            recommendations.append(
                f"🚨 Priority: Address {top_food['food']} (avg waste: {top_food['avg_waste_pct']}%). "
                f"Consider portion reduction or menu replacement."
            )
    
    recommendations.append("💡 Implement 'start small, come back' signage at serving stations.")
    recommendations.append("📊 Survey students on portion preferences for high-waste items.")
    recommendations.append("♻️ Share weekly waste data with students to increase awareness.")
    
    return recommendations


def assign_badge(avg_waste_pct: float) -> Dict:
    """Assign gamification badge based on waste performance."""
    if avg_waste_pct <= 10:
        return {"level": "Platinum", "emoji": "🏆", "description": "Zero-Waste Champion"}
    elif avg_waste_pct <= 20:
        return {"level": "Gold", "emoji": "🥇", "description": "Eco Warrior"}
    elif avg_waste_pct <= 35:
        return {"level": "Silver", "emoji": "🥈", "description": "Planet Protector"}
    elif avg_waste_pct <= 50:
        return {"level": "Bronze", "emoji": "🥉", "description": "Getting There"}
    else:
        return {"level": "Beginner", "emoji": "🌱", "description": "Room to Grow"}


def get_next_goal(current_points: int) -> Dict:
    """Get next achievement goal for gamification."""
    milestones = [
        (50, "Waste Warrior", "50 points"),
        (100, "Eco Champion", "100 points"),
        (250, "Planet Saver", "250 points"),
        (500, "Sustainability Hero", "500 points"),
        (1000, "Zero-Waste Legend", "1000 points")
    ]
    
    for points, title, desc in milestones:
        if current_points < points:
            return {
                "points_needed": points - current_points,
                "next_badge": title,
                "at": desc
            }
    
    return {"message": "Max level reached!", "next_badge": "Legend Status"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "dining-waste-tracker",
        "gemini_enabled": gemini_model is not None
    }


@app.get("/api/leaderboard")
async def get_leaderboard(school_id: str = "school_001", period: str = "week"):
    """
    Get student leaderboard for gamification.
    period: 'week', 'month', 'all'
    """
    if period == "week":
        cutoff = datetime.now() - timedelta(days=7)
    elif period == "month":
        cutoff = datetime.now() - timedelta(days=30)
    else:
        cutoff = datetime.min
    
    # Aggregate by student
    student_stats = defaultdict(lambda: {"points": 0, "scans": 0, "waste": 0})
    
    for scan in scans_db:
        if scan["school_id"] == school_id and datetime.fromisoformat(scan["timestamp"]) > cutoff:
            sid = scan.get("student_id")
            if sid:
                student_stats[sid]["points"] += scan["points"]
                student_stats[sid]["scans"] += 1
                student_stats[sid]["waste"] += scan["avg_waste_percentage"]
    
    # Create leaderboard
    leaderboard = []
    for student_id, stats in student_stats.items():
        avg_waste = stats["waste"] / stats["scans"] if stats["scans"] > 0 else 0
        leaderboard.append({
            "student_id": student_id,
            "total_points": stats["points"],
            "scans": stats["scans"],
            "avg_waste_pct": round(avg_waste, 1),
            "badge": assign_badge(avg_waste)
        })
    
    leaderboard.sort(key=lambda x: x["total_points"], reverse=True)
    
    # Add rankings
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1
    
    return JSONResponse({
        "period": period,
        "leaderboard": leaderboard[:50]  # Top 50
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)