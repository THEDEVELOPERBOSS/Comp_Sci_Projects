import { useEffect, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { initializeDatabase } from "../db/database";
export default function HomeScreen() {
  const [databaseReady, setDatabaseReady] = useState(false); // tracks database readiness state
  useEffect(() => { // runs the effect when the component mounts
    async function setupDatabase() { // defines an asychronous function to organize intializtion steps
      try {
        await initializeDatabase(); // waits for SQLite tables to be intialized
        setDatabaseReady(true); // marks the database as ready only after intialization succeeds
      } catch (error) { // reports initialization failures instead of silently ignoring them
        console.error("Failed to intialize database", error);
      }
    }

    setupDatabase();
  }, []);
  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.greeting}>Your Progress</Text>
      <Text style={styles.heading}>Goal Tracker</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Today's Habits</Text>
        <Text style={styles.cardText}>Your daily habits will appear here.</Text>
        <Text>databaseReady</Text>
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
});
