import { useEffect, useState} from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { StyleSheet, Text, View } from "react-native";

import {
    type CompletionHistory,
    getCompletionHistory,
}   from "../db/database";

export default function HistoryScreen() {
    const [history, setHistory] = useState<CompletionHistory[]>([]);

    useEffect(() => {
    async function loadHistory() {
        const savedHistory = await getCompletionHistory();

        setHistory(savedHistory);
    }

    loadHistory();
}, []);
    return (
    <SafeAreaView style={styles.container}>
        <Text style={styles.heading}>Completion History</Text>
        {history.map((item) => ( // .map() loops throught every item in an array and returns something for each item
            <View 
                key={`${item.habit_id}-${item.completion_date}`}
                style={styles.historyItem}
                >
                <Text>{item.habit_name}</Text>
                <Text>{item.completion_date}</Text>
            </View>
        ))}
    </SafeAreaView>
    );
}
// Styles
// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#F7F7FB",
    padding: 24,
    paddingTop: 60,
  },
  heading: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#172554",
    marginBottom: 20,
  },
  historyItem: {
    backgroundColor: "#FFFFFF",
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#E2E8F0",
  },
});