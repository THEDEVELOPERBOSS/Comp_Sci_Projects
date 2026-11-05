
import { useState } from "react";
import {
  Alert,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";
import { router } from "expo-router";

import { createHabit } from "../db/database";

const weekdays = [
  { label: "Mon", value: 1 },
  { label: "Tue", value: 2 },
  { label: "Wed", value: 3 },
  { label: "Thu", value: 4 },
  { label: "Fri", value: 5 },
  { label: "Sat", value: 6 },
  { label: "Sun", value: 7 },
];

export default function CreateHabitScreen() {
  const [habitName, setHabitName] = useState("");
  const [selectedDays, setSelectedDays] = useState<number[]>([]);
  const [weeklyTarget, setWeeklyTarget] = useState(3);
  const [saving, setSaving] = useState(false);

  // Add or remove a weekday from the selection.
  function toggleDay(day: number) {
    setSelectedDays((currentDays) =>
      currentDays.includes(day)
        ? currentDays.filter((selected) => selected !== day)
        : [...currentDays, day]
    );
  }

  // Save the habit to SQLite.
  async function handleSaveHabit() {
    const trimmedName = habitName.trim();

    if (!trimmedName) {
      Alert.alert("Missing habit name", "Please enter a habit name.");
      return;
    }

    if (selectedDays.length === 0) {
      Alert.alert("Missing schedule", "Please select at least one weekday.");
      return;
    }

    if (weeklyTarget > selectedDays.length) {
      Alert.alert(
        "Invalid weekly target",
        "Your weekly target cannot exceed the number of selected days."
      );
      return;
    }

    try {
      setSaving(true);

      await createHabit(trimmedName, weeklyTarget, selectedDays);

      Alert.alert("Success", "Your habit has been saved!", [
        {
          text: "OK",
          onPress: () => router.back(),
        },
      ]);
    } catch (error) {
      console.error("Failed to save habit:", error);
      Alert.alert("Error", "Something went wrong while saving your habit.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Pressable onPress={() => router.back()}>
          <Text style={styles.backButton}>← Back</Text>
        </Pressable>

        <Text style={styles.heading}>Create a Habit</Text>
        <Text style={styles.description}>
          Build consistency by tracking something you want to accomplish.
        </Text>

        <View style={styles.section}>
          <Text style={styles.label}>Habit Name</Text>

          <TextInput
            style={styles.input}
            placeholder="e.g. Read for 30 minutes"
            value={habitName}
            onChangeText={setHabitName}
            maxLength={60}
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>Choose Your Days</Text>

          <View style={styles.daysContainer}>
            {weekdays.map((day) => {
              const isSelected = selectedDays.includes(day.value);

              return (
                <Pressable
                  key={day.value}
                  style={[
                    styles.dayButton,
                    isSelected && styles.selectedDay,
                  ]}
                  onPress={() => toggleDay(day.value)}
                >
                  <Text
                    style={[
                      styles.dayText,
                      isSelected && styles.selectedDayText,
                    ]}
                  >
                    {day.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>Weekly Target</Text>
          <Text style={styles.description}>
            How many times per week do you want to complete this habit?
          </Text>

          <View style={styles.targetContainer}>
            {[1, 2, 3, 4, 5, 6, 7].map((number) => (
              <Pressable
                key={number}
                style={[
                  styles.targetButton,
                  weeklyTarget === number && styles.selectedDay,
                ]}
                onPress={() => setWeeklyTarget(number)}
              >
                <Text
                  style={[
                    styles.dayText,
                    weeklyTarget === number && styles.selectedDayText,
                  ]}
                >
                  {number}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        <Pressable
          style={[styles.saveButton, saving && styles.disabledButton]}
          onPress={handleSaveHabit}
          disabled={saving}
        >
          <Text style={styles.saveButtonText}>
            {saving ? "Saving..." : "Save Habit"}
          </Text>
        </Pressable>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F5F7FB",
  },
  content: {
    padding: 24,
    paddingBottom: 40,
  },
  backButton: {
    fontSize: 16,
    color: "#2563EB",
    marginBottom: 24,
  },
  heading: {
    fontSize: 30,
    fontWeight: "bold",
    color: "#172554",
    marginBottom: 8,
  },
  description: {
    fontSize: 14,
    color: "#64748B",
    marginBottom: 16,
  },
  section: {
    marginTop: 24,
  },
  label: {
    fontSize: 18,
    fontWeight: "600",
    color: "#172554",
    marginBottom: 12,
  },
  input: {
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E2E8F0",
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
  },
  daysContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  dayButton: {
    width: 43,
    height: 43,
    borderRadius: 12,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E2E8F0",
    alignItems: "center",
    justifyContent: "center",
  },
  selectedDay: {
    backgroundColor: "#2563EB",
    borderColor: "#2563EB",
  },
  dayText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#172554",
  },
  selectedDayText: {
    color: "#FFFFFF",
  },
  targetContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  targetButton: {
    width: 38,
    height: 42,
    borderRadius: 10,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "#E2E8F0",
    alignItems: "center",
    justifyContent: "center",
  },
  saveButton: {
    backgroundColor: "#2563EB",
    padding: 18,
    borderRadius: 12,
    alignItems: "center",
    marginTop: 36,
  },
  disabledButton: {
    opacity: 0.5,
  },
  saveButtonText: {
    color: "#FFFFFF",
    fontSize: 17,
    fontWeight: "bold",
  },
});
