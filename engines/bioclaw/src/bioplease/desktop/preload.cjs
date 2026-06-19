const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('biopleaseDesktop', {
  pickFolder: () => ipcRenderer.invoke('bioplease:pick-folder'),
})
