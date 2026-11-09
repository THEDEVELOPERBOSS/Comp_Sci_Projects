import { useEffect, useState } from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { initializeDatabase, getHabits, completeHabit, type Habit,} from "../../db/database";
import { router } from "expo-router";
import { StyleSheet, Text, View, Pressable } from "react-native";
export default function HomeScreen() {
  const [habits, setHabits] = useState<Habit[]>([]); // Habit holds the retrieved habit records
  // useState<Habit[]>([]) tells typescript that this will be an array of Habit objects
  // setHabits allows for udpating that array when SQLite returns the data

  useEffect(() => {
    async function setupDatabase() {
      try {
        // First, make sure the database tables exist.
        await initializeDatabase();

        // Retrieve all active habits from SQLite.
        const savedHabits = await getHabits();

        // Store the retrieved habits in React state.
        setHabits(savedHabits);


      } catch (error) {
        console.error("Failed to initialize database:", error);
      }
    }

    setupDatabase();
  }, []);

  async function handleCompleteHabit(habitId: number) {
    try {
      // Save todays completioon in SQLite
      await completeHabit(habitId);

      // Retrieve the updated habit records
      const updatedHabits = await getHabits();

      // Update the dashboard
      setHabits(updatedHabits);

    } catch (error) {
      console.error("Failed to completete habit:", error);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.greeting}>Your Progress</Text>
      <Text style={styles.heading}>Goal Tracker</Text>
      <Pressable
        style={styles.addButton}
        onPress={() => router.push("/create-habit")}
      >
        <Text style={styles.addButtonText}>+ Add Habit</Text>
      </Pressable>
      <Pressable
        style={styles.historyButton}
        onPress={() => router.push("/history")}
      >
        <Text style={styles.historyButtonText}>View Completion History</Text>
      </Pressable>
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Daily Habits</Text>

        {habits.length === 0 ? (
          <Text style={styles.cardText}>
            No habits yet. Press Add Habit to create one.
          </Text>
        ) : (
          habits.map((habit) => (
            <View key={habit.id} style={styles.habitItem}>
              <Text style={styles.habitName}>{habit.name}</Text>

              <Text style={styles.cardText}>
                {habit.weekly_target} times per week
              </Text>

              <Text style={styles.cardText}>
                Scheduled days: {habit.scheduled_days}
              </Text>
              <Pressable
                style={styles.completeButton}
                onPress={() => handleCompleteHabit(habit.id)}
                disabled={habit.completed_today === 1} // makes it so that once it is completed it can not be clicked again for that day 
              >
                <Text style={styles.completeButtonText}>
                  {habit.completed_today === 1 ? "Completed" : "Mark Complete"}
                </Text>
              </Pressable>
            </View>
          ))
        )}
      </View>


      <View style={styles.card}>
        <Text style={styles.cardTitle}>Your Goals</Text>
        <Text style={styles.cardText}>
          Track your goals and milestones here.
        </Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Progress Insights</Text>
        <Text style={styles.cardText}>
          Your personal statistics and AI predictions will appear here.
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F5F7FB",
    padding: 24,
    paddingTop: 60,
  },
  greeting: {
    fontSize: 16,
    color: "#64748B",
    padding: 10,
  },
  heading: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#172554",
    marginBottom: 28,
    padding: 10,
  },
  addButton: {
  backgroundColor: "#2563EB",
  padding: 16,
  borderRadius: 12,
  alignItems: "center",
  marginBottom: 20,
  },
  addButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "bold",
  },
  card: {
    backgroundColor: "#FFFFFF",
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: "#E2E8F0",
  },
  cardTitle: {
    fontSize: 19,
    fontWeight: "600",
    marginBottom: 8,
    color: "#172554",
  },
  cardText: {
    fontSize: 15,
    color: "#64748B",
  },
  habitItem: {
  paddingVertical: 12,
  borderBottomWidth: 1,
  borderBottomColor: "#E2E8F0",
  },

  habitName: {
    fontSize: 17,
    fontWeight: "600",
    color: "#172554",
    marginBottom: 6,
  },
  completeButton: {
    backgroundColor: "#00ff00",
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 12,
  },
  completeButtonText: {
    color: "#FFFFFF",
    fontWeight: "bold",
  },
  historyButton: {
    backgroundColor: "#172554",
    padding: 16, 
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 20,
  },
  historyButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "bold",
  },
});
