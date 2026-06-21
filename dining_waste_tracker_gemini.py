from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
import glob
import sqlite3
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
menu_logs_db = {}
FOODPRINT_DB_PATH = os.getenv(
    "FOODPRINT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "foodprint.db"),
)
DINING_HALLS = [
    "Crossroads",
    "Foothill",
]

# Constants
WASTE_LEVELS = {
    0.0: "None",
    0.1: "Minimal",
    0.25: "Moderate",
    0.40: "Significant",
    1.0: "Most Left"
}


class MealLog(BaseModel):
    items: List[str] = Field(default_factory=list)


class MenuLogRequest(BaseModel):
    school_id: str = "school_001"
    dining_hall: str = "Crossroads"
    date: str
    breakfast: MealLog = Field(default_factory=MealLog)
    lunch: MealLog = Field(default_factory=MealLog)
    dinner: MealLog = Field(default_factory=MealLog)


def normalize_menu_items(items: List[str]) -> List[str]:
    return sorted({item.strip() for item in items if item and item.strip()})


def infer_meal_from_timestamp(timestamp: str) -> str:
    hour = datetime.fromisoformat(timestamp).hour
    if hour < 11:
        return "breakfast"
    if hour < 16:
        return "lunch"
    return "dinner"



def ensure_sample_data() -> None:
    """Provide a small in-memory dataset so the dashboard has something to render."""
    if scans_db:
        return

    sample_scans = [
        {
            "timestamp": "2026-06-18T08:20:00",
            "school_id": "school_001",
            "dining_hall": "Crossroads",
            "student_id": "s101",
            "food_items": [
                {"name": "Scrambled Eggs", "waste_percentage": 18, "estimated_weight_oz": 1.2, "category": "entree"},
                {"name": "Poblano Fries", "waste_percentage": 34, "estimated_weight_oz": 1.8, "category": "side"},
            ],
            "avg_waste_percentage": 26.0,
            "waste_level": "Moderate",
            "points": 8,
            "impact": {"weight_lbs": 0.19, "weight_oz": 3.0, "cost_usd": 1.05, "co2_kg": 0.38, "water_gallons": 4.8},
            "overall_assessment": "Breakfast sides are underperforming.",
            "suggestions": ["Reduce breakfast potato portions."],
            "before_image": "",
            "after_image": "",
        },
        {
            "timestamp": "2026-06-18T12:35:00",
            "school_id": "school_001",
            "dining_hall": "Crossroads",
            "student_id": "s102",
            "food_items": [
                {"name": "Halal Rosemary Chicken", "waste_percentage": 12, "estimated_weight_oz": 0.9, "category": "entree"},
                {"name": "Yemeni Beef Stew", "waste_percentage": 42, "estimated_weight_oz": 2.6, "category": "entree"},
                {"name": "Pita Bread", "waste_percentage": 9, "estimated_weight_oz": 0.4, "category": "side"},
            ],
            "avg_waste_percentage": 21.0,
            "waste_level": "Moderate",
            "points": 10,
            "impact": {"weight_lbs": 0.24, "weight_oz": 3.9, "cost_usd": 1.32, "co2_kg": 0.48, "water_gallons": 6.0},
            "overall_assessment": "Stew is drawing more leftovers than other lunch entrees.",
            "suggestions": ["Offer smaller ladles for heavier stews."],
            "before_image": "",
            "after_image": "",
        },
        {
            "timestamp": "2026-06-18T18:10:00",
            "school_id": "school_001",
            "dining_hall": "Crossroads",
            "student_id": "s103",
            "food_items": [
                {"name": "Tofu Thai Curry", "waste_percentage": 11, "estimated_weight_oz": 0.8, "category": "entree"},
                {"name": "Roma Pesto Pizza", "waste_percentage": 29, "estimated_weight_oz": 1.4, "category": "entree"},
                {"name": "Braised Bok Choy", "waste_percentage": 37, "estimated_weight_oz": 1.1, "category": "vegetable"},
            ],
            "avg_waste_percentage": 25.7,
            "waste_level": "Moderate",
            "points": 8,
            "impact": {"weight_lbs": 0.21, "weight_oz": 3.3, "cost_usd": 1.16, "co2_kg": 0.42, "water_gallons": 5.2},
            "overall_assessment": "Vegetable acceptance dips at dinner.",
            "suggestions": ["Pair greens with a stronger sauce option."],
            "before_image": "",
            "after_image": "",
        },
        {
            "timestamp": "2026-06-19T12:25:00",
            "school_id": "school_001",
            "dining_hall": "Cafe 3",
            "student_id": "s104",
            "food_items": [
                {"name": "Broccoli Cheddar Soup", "waste_percentage": 31, "estimated_weight_oz": 1.9, "category": "soup"},
                {"name": "Chicken Salad", "waste_percentage": 8, "estimated_weight_oz": 0.5, "category": "entree"},
            ],
            "avg_waste_percentage": 19.5,
            "waste_level": "Moderate",
            "points": 10,
            "impact": {"weight_lbs": 0.15, "weight_oz": 2.4, "cost_usd": 0.83, "co2_kg": 0.30, "water_gallons": 3.8},
            "overall_assessment": "Soup waste remains elevated.",
            "suggestions": ["Test smaller soup cups at lunch."],
            "before_image": "",
            "after_image": "",
        },
        {
            "timestamp": "2026-06-19T18:40:00",
            "school_id": "school_001",
            "dining_hall": "Foothill",
            "student_id": "s105",
            "food_items": [
                {"name": "Shrimp Pesto Alfredo Sauce", "waste_percentage": 46, "estimated_weight_oz": 2.8, "category": "entree"},
                {"name": "Penne Pasta", "waste_percentage": 17, "estimated_weight_oz": 1.0, "category": "entree"},
            ],
            "avg_waste_percentage": 31.5,
            "waste_level": "Significant",
            "points": 5,
            "impact": {"weight_lbs": 0.24, "weight_oz": 3.8, "cost_usd": 1.31, "co2_kg": 0.48, "water_gallons": 6.0},
            "overall_assessment": "Alfredo sauce is causing outsized dinner waste.",
            "suggestions": ["Offer marinara by default, alfredo by request."],
            "before_image": "",
            "after_image": "",
        },
    ]

    for index, scan in enumerate(sample_scans, start=1):
        scan["id"] = index
        scans_db.append(scan)

    menu_logs_db["school_001:Crossroads:2026-06-18"] = {
        "school_id": "school_001",
        "dining_hall": "Crossroads",
        "date": "2026-06-18",
        "breakfast": {"items": ["Scrambled Eggs", "Poblano Fries", "Spinach Tofu Tomato Scramble"]},
        "lunch": {"items": ["Halal Rosemary Chicken", "Yemeni Beef Stew", "Pita Bread"]},
        "dinner": {"items": ["Tofu Thai Curry", "Roma Pesto Pizza", "Braised Bok Choy"]},
        "updated_at": "2026-06-18T07:00:00",
    }
    menu_logs_db["school_001:Crossroads:2026-06-19"] = {
        "school_id": "school_001",
        "dining_hall": "Crossroads",
        "date": "2026-06-19",
        "breakfast": {"items": ["Oatmeal", "Plain Bagels"]},
        "lunch": {"items": ["Broccoli Cheddar Soup", "Chicken Salad"]},
        "dinner": {"items": ["Shrimp Pesto Alfredo Sauce", "Penne Pasta"]},
        "updated_at": "2026-06-19T07:00:00",
    }
    menu_logs_db["school_001:Foothill:2026-06-19"] = {
        "school_id": "school_001",
        "dining_hall": "Foothill",
        "date": "2026-06-19",
        "breakfast": {"items": ["Steel Cut Oats", "Blueberry Muffins"]},
        "lunch": {"items": ["Chicken Shawarma", "Roasted Potatoes"]},
        "dinner": {"items": ["Shrimp Pesto Alfredo Sauce", "Penne Pasta"]},
        "updated_at": "2026-06-19T07:00:00",
    }


def find_latest_scraped_menu(location: str) -> Optional[str]:
    slug = location.lower().replace(" ", "-")
    menu_dir = os.path.join(os.path.dirname(__file__), "menus")
    matches = glob.glob(os.path.join(menu_dir, f"{slug}*.json"))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def menu_log_key(school_id: str, dining_hall: str, date: str) -> str:
    return f"{school_id}:{dining_hall}:{date}"


def get_foodprint_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(FOODPRINT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def estimate_item_waste_oz(relative_amount_pct: float) -> float:
    # Lightweight proxy until the watcher stores true weights.
    return max(0.4, (relative_amount_pct / 100.0) * 4.0)


def load_db_plate_events(dining_hall: str, days: int) -> List[Dict]:
    if not os.path.exists(FOODPRINT_DB_PATH):
        return []

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat()
    with get_foodprint_connection() as conn:
        events = conn.execute(
            """
            SELECT id, captured_at, dining_hall, capture_path, detection_confidence,
                   detection_reason, analysis_summary, analysis_json
            FROM plate_events
            WHERE dining_hall = ? AND captured_at >= ?
            ORDER BY captured_at DESC
            """,
            (dining_hall, cutoff_iso),
        ).fetchall()

        if not events:
            return []

        item_rows = conn.execute(
            """
            SELECT event_id, food_name, relative_amount_label, relative_amount_pct, notes
            FROM leftover_items
            WHERE event_id IN (
                SELECT id FROM plate_events WHERE dining_hall = ? AND captured_at >= ?
            )
            ORDER BY event_id
            """,
            (dining_hall, cutoff_iso),
        ).fetchall()

    items_by_event: Dict[int, List[Dict]] = defaultdict(list)
    for row in item_rows:
        items_by_event[row["event_id"]].append(
            {
                "name": row["food_name"],
                "waste_percentage": float(row["relative_amount_pct"]),
                "estimated_weight_oz": estimate_item_waste_oz(float(row["relative_amount_pct"])),
                "category": row["relative_amount_label"],
                "notes": row["notes"],
            }
        )

    scans = []
    for event in events:
        food_items = items_by_event.get(event["id"], [])
        total_weight_oz = sum(item["estimated_weight_oz"] for item in food_items)
        avg_waste_pct = (
            sum(item["waste_percentage"] for item in food_items) / len(food_items)
            if food_items else 0.0
        )
        impact = {
            "weight_oz": round(total_weight_oz, 2),
            "weight_lbs": round(total_weight_oz / 16.0, 3),
            "cost_usd": round((total_weight_oz / 16.0) * 5.5, 2),
            "co2_kg": round((total_weight_oz / 16.0) * 2.0, 2),
            "water_gallons": round((total_weight_oz / 16.0) * 25.0, 1),
        }
        scans.append(
            {
                "id": int(event["id"]),
                "timestamp": event["captured_at"],
                "school_id": "school_001",
                "dining_hall": event["dining_hall"],
                "food_items": food_items,
                "avg_waste_percentage": round(avg_waste_pct, 2),
                "impact": impact,
                "overall_assessment": event["analysis_summary"],
                "suggestions": [],
            }
        )
    return scans


def count_db_plate_events(dining_hall: str, days: int) -> int:
    if not os.path.exists(FOODPRINT_DB_PATH):
        return 0

    cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat()
    with get_foodprint_connection() as conn:
        row = conn.execute(
            """
            SELECT COUNT(pe.id) AS event_count
            FROM plate_events pe
            WHERE pe.dining_hall = ? AND pe.captured_at >= ?
            """,
            (dining_hall, cutoff_iso),
        ).fetchone()
    return int(row["event_count"] if row and row["event_count"] is not None else 0)


def build_staff_insights_from_scans(recent_scans: List[Dict]) -> List[Dict]:
    if not recent_scans:
        return []

    insights = []
    food_waste = defaultdict(lambda: {"waste": [], "count": 0})
    for scan in recent_scans:
        for item in scan.get("food_items", []):
            food_name = item.get("name", "Unknown")
            food_waste[food_name]["waste"].append(item.get("waste_percentage", 0))
            food_waste[food_name]["count"] += 1

    if food_waste:
        worst_food = max(food_waste.items(), key=lambda x: np.mean(x[1]["waste"]))
        avg_waste = np.mean(worst_food[1]["waste"])
        insights.append({
            "type": "alert" if avg_waste > 35 else "info",
            "title": f"Highest leftover item: {worst_food[0]}",
            "description": f"{worst_food[0]} is averaging {int(avg_waste)}% left on plates across {worst_food[1]['count']} detections.",
            "priority": "high" if avg_waste > 35 else "medium",
        })

        best_food = min(food_waste.items(), key=lambda x: np.mean(x[1]["waste"]))
        insights.append({
            "type": "success",
            "title": f"Lowest leftover item: {best_food[0]}",
            "description": f"{best_food[0]} is averaging only {int(np.mean(best_food[1]['waste']))}% left on plates.",
            "priority": "low",
        })

    total_weight = sum(s["impact"]["weight_lbs"] for s in recent_scans)
    total_cost = sum(s["impact"]["cost_usd"] for s in recent_scans)
    insights.append({
        "type": "info",
        "title": "Observed waste impact",
        "description": f"{round(total_weight, 1)} lbs observed, costing about ${round(total_cost, 2)} in the selected window.",
        "priority": "info",
    })

    daily_waste = defaultdict(list)
    for scan in recent_scans:
        day = datetime.fromisoformat(scan["timestamp"]).strftime("%A")
        daily_waste[day].append(scan["avg_waste_percentage"])
    if daily_waste:
        best_day = min(daily_waste.items(), key=lambda x: np.mean(x[1]))
        insights.append({
            "type": "info",
            "title": f"Best day: {best_day[0]}",
            "description": f"{best_day[0]} is currently the lowest-waste day at {int(np.mean(best_day[1]))}% average leftovers.",
            "priority": "low",
        })

    return insights


def load_recent_plate_events(dining_hall: str, limit: int = 12) -> List[Dict]:
    if not os.path.exists(FOODPRINT_DB_PATH):
        return []

    with get_foodprint_connection() as conn:
        rows = conn.execute(
            """
            SELECT pe.id, pe.captured_at, pe.dining_hall, pe.capture_path,
                   pe.detection_confidence, pe.detection_reason, pe.analysis_summary,
                   li.food_name, li.relative_amount_label, li.relative_amount_pct
            FROM plate_events pe
            LEFT JOIN leftover_items li ON li.event_id = pe.id
            WHERE pe.dining_hall = ?
            ORDER BY pe.captured_at DESC, li.id ASC
            LIMIT ?
            """,
            (dining_hall, limit * 4),
        ).fetchall()

    events: Dict[int, Dict] = {}
    ordered_ids: List[int] = []
    for row in rows:
        event_id = int(row["id"])
        if event_id not in events:
            events[event_id] = {
                "id": event_id,
                "captured_at": row["captured_at"],
                "dining_hall": row["dining_hall"],
                "capture_path": row["capture_path"],
                "detection_confidence": float(row["detection_confidence"]),
                "detection_reason": row["detection_reason"],
                "analysis_summary": row["analysis_summary"],
                "food_items": [],
            }
            ordered_ids.append(event_id)
        if row["food_name"]:
            events[event_id]["food_items"].append(
                {
                    "name": row["food_name"],
                    "relative_amount_label": row["relative_amount_label"],
                    "relative_amount_pct": float(row["relative_amount_pct"]),
                }
            )

    return [events[event_id] for event_id in ordered_ids[:limit]]


def get_plate_event_capture_path(event_id: int) -> Optional[str]:
    if not os.path.exists(FOODPRINT_DB_PATH):
        return None
    with get_foodprint_connection() as conn:
        row = conn.execute(
            "SELECT capture_path FROM plate_events WHERE id = ?",
            (event_id,),
        ).fetchone()
    if not row:
        return None
    return row["capture_path"]


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
    school_id: str = "school_001",
    dining_hall: str = "Crossroads",
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
            "dining_hall": dining_hall,
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
async def get_insights(
    school_id: str = "school_001",
    days: int = 30,
    dining_hall: str = "Crossroads",
):
    """
    Get AI-generated insights and recommendations for dining staff.
    """
    recent_scans = load_db_plate_events(dining_hall, days)
    if not recent_scans:
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_scans = [
            s for s in scans_db
            if s["school_id"] == school_id
            and datetime.fromisoformat(s["timestamp"]) > cutoff_date
            and s.get("dining_hall", "Crossroads") == dining_hall
        ]
    
    if not recent_scans:
        return JSONResponse({"insights": []})
    return JSONResponse({"insights": build_staff_insights_from_scans(recent_scans)})


@app.get("/api/staff/menu-log")
async def get_menu_log(
    school_id: str = "school_001",
    dining_hall: str = "Crossroads",
    date: Optional[str] = None,
):
    ensure_sample_data()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    key = menu_log_key(school_id, dining_hall, date)
    record = menu_logs_db.get(key)
    if not record:
        record = {
            "school_id": school_id,
            "date": date,
            "dining_hall": dining_hall,
            "breakfast": {"items": []},
            "lunch": {"items": []},
            "dinner": {"items": []},
            "updated_at": None,
        }
    return JSONResponse(record)


@app.post("/api/staff/menu-log")
async def save_menu_log(payload: MenuLogRequest):
    ensure_sample_data()
    key = menu_log_key(payload.school_id, payload.dining_hall, payload.date)
    record = {
        "school_id": payload.school_id,
        "dining_hall": payload.dining_hall,
        "date": payload.date,
        "breakfast": {"items": normalize_menu_items(payload.breakfast.items)},
        "lunch": {"items": normalize_menu_items(payload.lunch.items)},
        "dinner": {"items": normalize_menu_items(payload.dinner.items)},
        "updated_at": datetime.now().isoformat(),
    }
    menu_logs_db[key] = record
    return JSONResponse({"success": True, "menu_log": record})


@app.post("/api/staff/menu-log/import-scraped")
async def import_scraped_menu_log(
    school_id: str = "school_001",
    dining_hall: str = "Crossroads",
    date: Optional[str] = None,
):
    ensure_sample_data()
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    scraped_path = find_latest_scraped_menu(dining_hall)
    if not scraped_path:
        raise HTTPException(status_code=404, detail=f"No scraped menu file found for {dining_hall}.")

    try:
        with open(scraped_path, "r", encoding="utf-8") as f:
            scraped = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read scraped menu file: {exc}")

    record = {
        "school_id": school_id,
        "dining_hall": scraped.get("location", dining_hall),
        "date": date,
        "breakfast": {"items": normalize_menu_items(scraped.get("breakfast", []))},
        "lunch": {"items": normalize_menu_items(scraped.get("lunch", []))},
        "dinner": {"items": normalize_menu_items(scraped.get("dinner", []))},
        "updated_at": datetime.now().isoformat(),
        "imported_from": os.path.basename(scraped_path),
        "source_date_label": scraped.get("dateLabel"),
    }
    menu_logs_db[menu_log_key(school_id, record["dining_hall"], date)] = record
    return JSONResponse({"success": True, "menu_log": record})


@app.get("/api/staff/overview")
async def get_staff_overview(
    school_id: str = "school_001",
    days: int = 14,
    dining_hall: str = "Crossroads",
):
    db_event_count = count_db_plate_events(dining_hall, days)
    using_real_data = db_event_count > 0
    recent_scans = load_db_plate_events(dining_hall, days) if using_real_data else []
    if not recent_scans:
        ensure_sample_data()
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_scans = [
            s for s in scans_db
            if s["school_id"] == school_id
            and datetime.fromisoformat(s["timestamp"]) >= cutoff_date
            and s.get("dining_hall", "Crossroads") == dining_hall
        ]

    totals = {
        "total_scans": len(recent_scans),
        "weight_lbs": round(sum(s["impact"]["weight_lbs"] for s in recent_scans), 2),
        "cost_usd": round(sum(s["impact"]["cost_usd"] for s in recent_scans), 2),
        "co2_kg": round(sum(s["impact"]["co2_kg"] for s in recent_scans), 2),
        "avg_waste_pct": round(
            sum(s["avg_waste_percentage"] for s in recent_scans) / len(recent_scans), 1
        ) if recent_scans else 0.0,
    }

    meal_buckets = {
        "breakfast": {"scans": 0, "waste_values": [], "weight_lbs": 0.0},
        "lunch": {"scans": 0, "waste_values": [], "weight_lbs": 0.0},
        "dinner": {"scans": 0, "waste_values": [], "weight_lbs": 0.0},
    }
    daily_buckets = defaultdict(lambda: {"scans": 0, "waste_values": [], "weight_lbs": 0.0, "cost_usd": 0.0})
    food_buckets = defaultdict(lambda: {"appearances": 0, "waste_values": [], "weight_oz": 0.0, "category": "other"})

    for scan in recent_scans:
        meal = infer_meal_from_timestamp(scan["timestamp"])
        meal_buckets[meal]["scans"] += 1
        meal_buckets[meal]["waste_values"].append(scan["avg_waste_percentage"])
        meal_buckets[meal]["weight_lbs"] += scan["impact"]["weight_lbs"]

        day_key = datetime.fromisoformat(scan["timestamp"]).strftime("%Y-%m-%d")
        daily_buckets[day_key]["scans"] += 1
        daily_buckets[day_key]["waste_values"].append(scan["avg_waste_percentage"])
        daily_buckets[day_key]["weight_lbs"] += scan["impact"]["weight_lbs"]
        daily_buckets[day_key]["cost_usd"] += scan["impact"]["cost_usd"]

        for item in scan.get("food_items", []):
            bucket = food_buckets[item.get("name", "Unknown")]
            bucket["appearances"] += 1
            bucket["waste_values"].append(item.get("waste_percentage", 0))
            bucket["weight_oz"] += item.get("estimated_weight_oz", 0)
            bucket["category"] = item.get("category", "other")

    meal_breakdown = []
    for meal_name, stats in meal_buckets.items():
        meal_breakdown.append({
            "meal": meal_name,
            "scans": stats["scans"],
            "avg_waste_pct": round(sum(stats["waste_values"]) / len(stats["waste_values"]), 1) if stats["waste_values"] else 0.0,
            "weight_lbs": round(stats["weight_lbs"], 2),
        })

    daily_trends = []
    for day_key in sorted(daily_buckets.keys()):
        stats = daily_buckets[day_key]
        daily_trends.append({
            "date": day_key,
            "scans": stats["scans"],
            "avg_waste_pct": round(sum(stats["waste_values"]) / len(stats["waste_values"]), 1) if stats["waste_values"] else 0.0,
            "weight_lbs": round(stats["weight_lbs"], 2),
            "cost_usd": round(stats["cost_usd"], 2),
        })

    top_waste_foods = []
    for name, stats in food_buckets.items():
        avg_waste = sum(stats["waste_values"]) / len(stats["waste_values"]) if stats["waste_values"] else 0.0
        top_waste_foods.append({
            "food": name,
            "category": stats["category"],
            "appearances": stats["appearances"],
            "avg_waste_pct": round(avg_waste, 1),
            "total_wasted_oz": round(stats["weight_oz"], 2),
        })
    top_waste_foods.sort(key=lambda item: item["avg_waste_pct"], reverse=True)

    insights_payload = json.loads(
        (await get_insights(school_id=school_id, days=days, dining_hall=dining_hall)).body.decode()
    )

    return JSONResponse({
        "school_id": school_id,
        "dining_hall": dining_hall,
        "dining_halls": DINING_HALLS,
        "days": days,
        "using_real_data": using_real_data,
        "real_event_count": db_event_count,
        "totals": totals,
        "meal_breakdown": meal_breakdown,
        "daily_trends": daily_trends,
        "top_waste_foods": top_waste_foods[:8],
        "insights": insights_payload["insights"],
    })


@app.get("/api/staff/recent-events")
async def get_recent_events(
    school_id: str = "school_001",
    dining_hall: str = "Crossroads",
    limit: int = 10,
):
    del school_id
    events = load_recent_plate_events(dining_hall, limit=max(1, min(limit, 25)))
    return JSONResponse(
        {
            "dining_hall": dining_hall,
            "events": events,
        }
    )


@app.get("/api/staff/recent-events/{event_id}/image")
async def get_recent_event_image(event_id: int):
    capture_path = get_plate_event_capture_path(event_id)
    if not capture_path:
        raise HTTPException(status_code=404, detail="Plate event image not found.")
    resolved = os.path.abspath(capture_path)
    workspace_root = os.path.abspath(os.path.dirname(__file__))
    if not resolved.startswith(workspace_root):
        raise HTTPException(status_code=403, detail="Image path is outside the project.")
    if not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail="Image file is missing.")
    return FileResponse(resolved)


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
    ensure_sample_data()
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


@app.get("/staff", response_class=HTMLResponse)
async def staff_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "staff_dashboard.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
