/** @type {import('tailwindcss').Config} */
export default {
  // The 'content' array tells Tailwind which files to scan for class names.
  // We include common React file extensions (.js, .jsx) in the 'src' directory,
  // assuming your main files (like renderer.jsx and App.jsx) are located there.
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    // If your files are in the root directory and not 'src', you may also need:
    // "./*.{js,ts,jsx,tsx}",
    "./**/*.{js,ts,jsx,tsx}", // Scan all directories recursively
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}