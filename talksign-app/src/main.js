import { app, BrowserWindow, Menu, ipcMain } from 'electron';
import path from 'node:path';
import started from 'electron-squirrel-startup';

if (started) {
  app.quit();
}

const createWindow = () => {
  const mainWindow = new BrowserWindow({
    width: 1100,
    height: 690,
    useContentSize: true, // <--- This ensures the inner area is exactly 1100x690
    resizable: true,
    autoHideMenuBar: true,
    frame: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  Menu.setApplicationMenu(null);

  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`));
  }
};

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// --- COMPACT MODE LISTENER ---
ipcMain.on('app:compact-mode', (event, isCompact) => {
  const win = BrowserWindow.fromWebContents(event.sender);
  if (win) {
    if (isCompact) {
      // Shrink to Overlay Strip
      win.setSize(1100, 100);
      win.setAlwaysOnTop(true, 'screen-saver'); 
    } else {
      // FIX: Use setContentSize so the inner area matches exactly 1100x690 again
      // This prevents the layout from squashing or looking "different"
      win.setContentSize(1100, 690);
      win.setAlwaysOnTop(false);
      win.center();
    }
  }
});