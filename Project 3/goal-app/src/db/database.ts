import * as SQLite from "expo-sqlite";
export type Habit = { // This is a TypeScript type 
  id: number; // describes what one habit looks like when SQLite returns it.
  name: string;
  created_at: string;
  active: number;
  weekly_target: number | null;
  scheduled_days: string | null;
};
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
export async function createHabit(
  name: string,
  weeklyTarget: number,
  selectedDays: number[]

) {
  const db = await dbPromise;

  // Insert the new habit
  const result = await db.runAsync(
    "INSERT INTO habits (name, weekly_target) VALUES (?, ?)", // Adds a new database record. ? Reprsents a value SQLite will safely insert
    [name, weeklyTarget]
  );

  // Get the ID SQLite assigned to the habit
  const habitId = result.lastInsertRowId; // retrieves the ID of the newly created habit

  // Save each selected weekday
  for (const day of selectedDays) { // loops through the selected weekdays
    await db.runAsync(
      "INSERT INTO habit_schedule_days (habit_id, weekday) VALUES (?, ?)",
      [habitId, day]
    );
  }
  return habitId;
}

export async function getHabits(): Promise<Habit[]> {
  const db = await dbPromise;

  const habits = await db.getAllAsync<Habit>(`
    SELECT
      habits.*,
      -- Combine all selected weekdays into one string.
      GROUP_CONCAT(habit_schedule_days.weekday) AS scheduled_days

    FROM habits

    -- Connect habits to their scheduled weekdays.
    LEFT JOIN habit_schedule_days 
      ON habits.id = habit_schedule_days.habit_id

    -- Only retrieve habits that are currently active.
    WHERE habits.active = 1

    -- Group schedule days together for each habit.
    GROUP BY habits.id

    -- Show the newest habits first.
    ORDER BY habits.created_at DESC
`);

  return habits;
}
