# Browserbase Menu Agent

This script uses Browserbase + Stagehand to open the Berkeley Dining menu site, select a dining hall and date, and extract breakfast, lunch, and dinner items into JSON.

## Install

```bash
npm install
```

## Environment

```bash
export BROWSERBASE_API_KEY="your_browserbase_api_key"
```

Optional:

```bash
export BROWSERBASE_MODEL="google/gemini-2.5-flash"
```

## Run

Crossroads for today:

```bash
npm run scrape:crossroads-menu
```

Print JSON too:

```bash
node browserbase_crossroads_menu_agent.mjs --stdout
```

Pick a different date label shown on the site:

```bash
node browserbase_crossroads_menu_agent.mjs --date "Tomorrow"
```

Pick a different hall:

```bash
node browserbase_crossroads_menu_agent.mjs --location "Café 3"
```

## Output

The script writes JSON files into `menus/`, for example:

```text
menus/crossroads-today.json
```

The JSON shape is:

```json
{
  "scrapedAt": "2026-06-20T00:00:00.000Z",
  "sourceUrl": "https://dining.berkeley.edu/menus/",
  "location": "Crossroads",
  "dateLabel": "Today",
  "breakfast": ["Item 1", "Item 2"],
  "lunch": ["Item 1", "Item 2"],
  "dinner": ["Item 1", "Item 2"]
}
```
