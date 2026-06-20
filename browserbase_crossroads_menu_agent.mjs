import { Stagehand } from "@browserbasehq/stagehand";
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";

const DEFAULT_URL = "https://dining.berkeley.edu/menus/";
const DEFAULT_LOCATION = "Crossroads";
const DEFAULT_MODEL = process.env.BROWSERBASE_MODEL || "google/gemini-2.5-flash";

function parseArgs(argv) {
  const args = {
    url: DEFAULT_URL,
    location: DEFAULT_LOCATION,
    dateLabel: "Today",
    model: DEFAULT_MODEL,
    outputDir: "menus",
    stdout: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];

    if (arg === "--url" && next) {
      args.url = next;
      i += 1;
    } else if (arg === "--location" && next) {
      args.location = next;
      i += 1;
    } else if (arg === "--date" && next) {
      args.dateLabel = next;
      i += 1;
    } else if (arg === "--model" && next) {
      args.model = next;
      i += 1;
    } else if (arg === "--output-dir" && next) {
      args.outputDir = next;
      i += 1;
    } else if (arg === "--stdout") {
      args.stdout = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }
  }

  return args;
}

function printHelp() {
  console.log(`
Usage:
  node browserbase_crossroads_menu_agent.mjs [options]

Options:
  --location "Crossroads"   Dining hall to scrape. Default: Crossroads
  --date "Today"            Date label shown on the site. Default: Today
  --model MODEL             Browserbase/Stagehand model. Default: ${DEFAULT_MODEL}
  --output-dir DIR          Where JSON output is written. Default: menus
  --stdout                  Also print the final JSON to stdout
  --url URL                 Override the menus page URL

Required env:
  BROWSERBASE_API_KEY
`);
}

const MealSchema = z.object({
  items: z
    .array(
      z
        .string()
        .min(1)
        .transform((value) => value.trim())
    )
    .default([]),
});

const MenuSchema = z.object({
  location: z.string(),
  dateLabel: z.string(),
  breakfast: MealSchema,
  lunch: MealSchema,
  dinner: MealSchema,
});

function dedupe(items) {
  return [...new Set(items.map((item) => item.trim()).filter(Boolean))];
}

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function main() {
  if (!process.env.BROWSERBASE_API_KEY) {
    throw new Error("Set BROWSERBASE_API_KEY before running this script.");
  }

  const args = parseArgs(process.argv.slice(2));
  const stagehand = new Stagehand({
    env: "BROWSERBASE",
    model: args.model,
  });

  await stagehand.init();

  try {
    const page = stagehand.context.pages()[0];
    await page.goto(args.url, { waitUntil: "domcontentloaded" });
    await page.waitForLoadState("networkidle").catch(() => {});

    await stagehand.act(
      `On the Berkeley Dining menus page, set the location to ${args.location} and set the date to ${args.dateLabel}. Ensure the page is showing the menu for ${args.location} with breakfast, lunch, and dinner sections for that date.`
    );

    const extracted = await stagehand.extract(
      `Extract the displayed ${args.location} daily menu. Return only food item names for breakfast, lunch, and dinner.
Do not include allergen labels, carbon labels, section names, hours, or duplicate items.
If a meal section is missing on the page, return an empty items array for that meal.`,
      MenuSchema,
      {
        screenshot: true,
        timeout: 120000,
      }
    );

    const result = {
      scrapedAt: new Date().toISOString(),
      sourceUrl: args.url,
      location: extracted.location || args.location,
      dateLabel: extracted.dateLabel || args.dateLabel,
      breakfast: dedupe(extracted.breakfast.items),
      lunch: dedupe(extracted.lunch.items),
      dinner: dedupe(extracted.dinner.items),
    };

    await ensureDir(args.outputDir);

    const safeLocation = result.location.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const safeDate = result.dateLabel.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const outputPath = path.join(args.outputDir, `${safeLocation}-${safeDate}.json`);

    await fs.writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`, "utf8");

    console.log(`Saved menu JSON to ${outputPath}`);
    console.log(
      `Breakfast: ${result.breakfast.length} items, Lunch: ${result.lunch.length} items, Dinner: ${result.dinner.length} items`
    );

    if (args.stdout) {
      console.log(JSON.stringify(result, null, 2));
    }
  } finally {
    await stagehand.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
