import * as SQLite from "expo-sqlite";

// Open a local database stored on the device.
export const dbPromise = SQLite.openDatabaseAsync("goal_tracker.db");

// Create the tables the app needs.
export async function initializeDatabase() {
  const db = await dbPromise;
  await db.execAsync("PRAGMA foreign_keys = ON;");
  await db.execAsync(`
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS habits (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      active INTEGER NOT NULL DEFAULT 1,
      weekly_target INTEGER CHECK (weekly_target BETWEEN 1 AND 7)
    );

    CREATE TABLE IF NOT EXISTS habit_completions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      habit_id INTEGER NOT NULL,
      completion_date TEXT NOT NULL,
      UNIQUE (habit_id, completion_date),
      FOREIGN KEY (habit_id) REFERENCES habits(id)
        ON DELETE CASCADE
    );
    -- SQLite generally expects all column definitions first, 
    -- followed by table-level constraints like PRIMARY KEY and FOREIGN KEY
    CREATE TABLE IF NOT EXISTS habit_schedule_days ( 
      habit_id INTEGER NOT NULL,
      weekday INTEGER NOT NULL CHECK (weekday BETWEEN 1 AND 7),
      PRIMARY KEY (habit_id, weekday),
      FOREIGN KEY (habit_id) REFERENCES habits(id)
        ON DELETE CASCADE 
    );
  `);
}
