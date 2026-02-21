import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import { ICommandPalette } from '@jupyterlab/apputils';
import { INotebookTracker } from '@jupyterlab/notebook';

import { ContextChatPanel } from './panel';

const PLUGIN_ID = 'context-chat:plugin';

/**
 * JupyterLab extension that adds a context-aware chat sidebar.
 * 
 * This extension provides a chat interface that can:
 * - Read context from the active notebook (selected cell or entire notebook)
 * - Send messages to a Python server endpoint with LangChain integration
 * - Display AI responses in a sidebar panel
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: PLUGIN_ID,
  autoStart: true,
  requires: [ICommandPalette, INotebookTracker],
  activate: (app: JupyterFrontEnd, palette: ICommandPalette, notebooks: INotebookTracker) => {
    console.log('JupyterLab extension context-chat is activated!');
    
    const { shell, commands } = app;
    let panel: ContextChatPanel | null = null;

    // Add command to open the chat panel
    commands.addCommand('context-chat:open', {
      label: 'Open Context Chat',
      caption: 'Open AI chat with notebook context awareness',
      execute: () => {
        // Create panel if it doesn't exist or was disposed
        if (!panel || panel.isDisposed) {
          panel = new ContextChatPanel(notebooks);
          panel.id = 'context-chat-panel';
          panel.title.caption = 'Context Chat';
          panel.title.label = 'Context Chat';
          panel.title.closable = true;
        }
        
        // Add to shell if not already attached
        if (!panel.isAttached) {
          shell.add(panel, 'left', { rank: 800 });
        }
        
        // Activate the panel
        shell.activateById(panel.id);
      }
    });

    // Add command to clear chat history
    commands.addCommand('context-chat:clear', {
      label: 'Clear Chat History',
      caption: 'Clear the conversation history',
      execute: () => {
        if (panel && !panel.isDisposed) {
          panel.clearChat();
        }
      }
    });

    // Add commands to palette
    palette.addItem({ command: 'context-chat:open', category: 'AI Assistant' });
    palette.addItem({ command: 'context-chat:clear', category: 'AI Assistant' });

    console.log('Context Chat commands registered');
  }
};

export default plugin;
