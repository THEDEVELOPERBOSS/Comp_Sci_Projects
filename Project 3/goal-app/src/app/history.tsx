import { useEffect, useState} from "react";
import { SafeAreaView } from "react-native-safe-area-context";
import { StyleSheet, Text, View } from "react-native";

import {
    type CompletionHistory,
    getCompletionHistory,
}   from "../db/database";

export default function HistoryScreen() {
    const [history, setHistory] = useState<CompletionHistory[]>([]);

    const groupedHistory = history.reduce((groups, item) => { // loops through every completion in the history
    const date = item.completion_date; 

    if (!groups[date]) { // checks whether a group has been created for that date
        // if none if found an empty array is created
        groups[date] = [];
    }

    groups[date].push(item); // adds current completion to the respective dates array

    return groups; // passes the updated groups to the next iteration
    }, {} as Record<string, CompletionHistory[]>);

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
        {Object.entries(groupedHistory).map(([date, items]) => (
            <View key={date}>
                <Text style={styles.dateHeading}>{date}</Text>

                {items.map((item) => (
                    <View
                        key={`${item.habit_id}-${item.completion_date}`}
                        style={styles.historyItem}
                    >
                        <Text>{item.habit_name}</Text>
                    </View>
                ))}
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
    dateHeading: {
        fontSize: 18,
        fontWeight: "bold",
        color: "#172554",
        marginBottom: 12,
        marginTop: 12,
    },
});