#!/usr/bin/env bash

set -u

PROJECT_DIR="$(pwd)"
LOG_FILE="$PROJECT_DIR/expo-repair.log"

exec > >(tee "$LOG_FILE") 2>&1

echo "========================================"
echo " Expo Project Diagnostic and Repair"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Log: $LOG_FILE"
echo

# 1. Confirm we're in the correct project
if [[ ! -f package.json ]]; then
    echo "ERROR: package.json not found."
    echo "Run this script from your Expo project directory."
    exit 1
fi

# 2. Back up important configuration files
BACKUP_DIR="$PROJECT_DIR/.expo-repair-backup/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

for file in package.json package-lock.json app.json app.config.js app.config.ts tsconfig.json; do
    if [[ -f "$file" ]]; then
        cp -p "$file" "$BACKUP_DIR/"
    fi
done

echo "[OK] Configuration backups created: $BACKUP_DIR"
echo

# 3. Check installed tools
echo "=== Checking Node and npm ==="
node --version || true
npm --version || true
echo

# 4. Check project dependencies
echo "=== Checking Expo dependencies ==="
npx expo-doctor || echo "[WARNING] Expo Doctor reported problems."
echo

# 5. Verify package.json can be parsed
echo "=== Validating package.json ==="
if node -e "JSON.parse(require('fs').readFileSync('package.json', 'utf8'))"; then
    echo "[OK] package.json is valid JSON."
else
    echo "[ERROR] package.json is invalid. Stopping to avoid unsafe repairs."
    exit 1
fi
echo

# 6. Check common Expo configuration files
echo "=== Checking Expo configuration ==="

if [[ -f app.json ]]; then
    if node -e "JSON.parse(require('fs').readFileSync('app.json', 'utf8'))"; then
        echo "[OK] app.json is valid JSON."
    else
        echo "[ERROR] app.json contains invalid JSON."
    fi
fi

if [[ -f app.config.js || -f app.config.ts ]]; then
    echo "[INFO] Dynamic Expo configuration detected."
    echo "Expo Doctor will help validate the resolved configuration."
fi
echo

# 7. Check common asset paths in app.json
echo "=== Checking configured assets ==="

node <<'NODE'
const fs = require('fs');
const path = require('path');

if (!fs.existsSync('app.json')) {
    console.log('[INFO] No app.json found; configuration may be dynamic.');
    process.exit(0);
}

const config = JSON.parse(fs.readFileSync('app.json', 'utf8'));
const expo = config.expo || config;

const candidates = [
    ['icon', expo.icon],
    ['splash.image', expo.splash?.image],
    ['android.adaptiveIcon.foregroundImage', expo.android?.adaptiveIcon?.foregroundImage],
    ['ios.icon', expo.ios?.icon],
];

for (const [name, value] of candidates) {
    if (typeof value !== 'string') continue;

    // Remote assets are not checked as local files.
    if (/^https?:\/\//i.test(value)) {
        console.log(`[INFO] ${name}: remote URL; not checked locally.`);
        continue;
    }

    const assetPath = path.resolve(process.cwd(), value);

    if (fs.existsSync(assetPath)) {
        console.log(`[OK] ${name}: ${value}`);
    } else {
        console.log(`[WARNING] ${name}: missing asset ${value}`);
    }
}
NODE

echo

# 8. Check Expo Router entry point
echo "=== Checking Expo Router ==="

if [[ -d src/app ]]; then
    echo "[OK] Found src/app."
elif [[ -d app ]]; then
    echo "[OK] Found app/."
else
    echo "[WARNING] Could not find app/ or src/app/."
fi

node -e '
const p = require("./package.json");
console.log("Entry point:", p.main || "(Expo default)");
console.log("Expo dependency:", p.dependencies?.expo || "(not declared)");
console.log("Expo Router:", p.dependencies?.["expo-router"] || "(not declared)");
' || true

echo

# 9. Check installed packages without deleting node_modules
echo "=== Checking installed dependencies ==="

if [[ -d node_modules ]]; then
    echo "[OK] node_modules exists."
else
    echo "[WARNING] node_modules is missing."
    echo "Attempting npm install..."
    npm install || echo "[ERROR] npm install failed."
fi

echo

# 10. Attempt safe dependency alignment
echo "=== Aligning Expo package versions ==="
npx expo install --check || {
    echo "[WARNING] Version mismatches detected or unable to check."
    echo "Attempting Expo's automatic package alignment..."
    npx expo install --fix || echo "[WARNING] Automatic alignment failed."
}

echo

# 11. Re-check the project
echo "=== Final Expo Doctor check ==="
npx expo-doctor || echo "[WARNING] Review the diagnostic output above."

echo
echo "========================================"
echo " Diagnostics and safe repairs finished."
echo "========================================"
echo "Log saved to: $LOG_FILE"
echo
echo "Next, start Expo with:"
echo "npx expo start --tunnel --clear"
echo
echo "If the manifest asset warning remains, inspect"
echo "expo-repair.log and the asset paths in app.json."
