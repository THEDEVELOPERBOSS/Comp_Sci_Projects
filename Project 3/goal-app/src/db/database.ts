
import * as SQLite from 'expo-sqlite';

// Open a local database stored on the device.
export const dbPromise = SQLite.openDatabaseAsync('goal_tracker.db');

// Create the tables the app needs.
export async function initializeDatabase() {
  const db = await dbPromise;

  await db.execAsync(`
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS habits (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      active INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS habit_completions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      habit_id INTEGER NOT NULL,
      completion_date TEXT NOT NULL,
      UNIQUE (habit_id, completion_date),
      FOREIGN KEY (habit_id) REFERENCES habits(id)
        ON DELETE CASCADE
    );
  `);

  await db.execAsync('PRAGMA foreign_keys = ON;');
}
