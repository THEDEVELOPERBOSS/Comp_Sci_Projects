import { useColorScheme as useRNColorScheme } from 'react-native';

/**
 * Use the native color scheme on the client and keep a safe default for SSR/static rendering.
 */
export function useColorScheme() {
  const colorScheme = useRNColorScheme();

  return colorScheme ?? 'light';
}
