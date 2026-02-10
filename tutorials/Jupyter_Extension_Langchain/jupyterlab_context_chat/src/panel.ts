import { Widget } from '@lumino/widgets';
import { INotebookTracker } from '@jupyterlab/notebook';
import { ServerConnection } from '@jupyterlab/services';

/**
 * Extract context from the active notebook.
 * 
 * Strategy:
 * 1. If there's an active cell, send its content
 * 2. Otherwise, send the entire notebook source (truncated if too large)
 * 
 * This is minimal and safe - we only send source code, not runtime variables.
 */
function getActiveNotebookContext(notebooks: INotebookTracker): string {
  const current = notebooks.currentWidget;
  if (!current) {
    return 'No active notebook.';
  }

  const nb = current.content;
  const activeCell = nb.activeCell;

  // Strategy 1: Send active cell content
  if (activeCell) {
    const cellText = activeCell.model.sharedModel.getSource() ?? '';
    if (cellText.trim()) {
      const cellType = activeCell.model.type;
      return `Active cell (${cellType}):\n\`\`\`\n${cellText}\n\`\`\``;
    }
  }

  // Strategy 2: Send entire notebook source
  const cells = Array.from(nb.widgets);
  const cellContents = cells.map((cell, idx) => {
    const cellType = cell.model.type;
    const content = cell.model.sharedModel.getSource();
    return `Cell ${idx + 1} (${cellType}):\n\`\`\`\n${content}\n\`\`\``;
  });
  
  const joined = cellContents.join('\n\n---\n\n');

  // Prevent huge payloads
  const MAX_CHARS = 30_000;
  if (joined.length > MAX_CHARS) {
    return joined.slice(0, MAX_CHARS) + '\n\n[Context truncated due to size...]';
  }
  
  return joined;
}

/**
 * Make a POST request to a server endpoint with JSON body.
 */
async function postJSON<T>(path: string, body: any): Promise<T> {
  const settings = ServerConnection.makeSettings();
  const url = new URL(path, settings.baseUrl).toString();

  const resp = await ServerConnection.makeRequest(
    url,
    {
      method: 'POST',
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json' }
    },
    settings
  );

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Server error ${resp.status}: ${text}`);
  }
  
  return resp.json() as Promise<T>;
}

/**
 * The main chat panel widget.
 * 
 * This creates a sidebar panel with:
 * - A scrollable conversation log
 * - A text input area
 * - A send button
 */
export class ContextChatPanel extends Widget {
  private _input: HTMLTextAreaElement;
  private _send: HTMLButtonElement;
  private _clear: HTMLButtonElement;
  private _log: HTMLDivElement;
  private _notebooks: INotebookTracker;
  private _contextToggle: HTMLInputElement;
  private _contextLabel: HTMLLabelElement;

  constructor(notebooks: INotebookTracker) {
    super();
    this._notebooks = notebooks;

    // Setup panel styles
    this.node.style.padding = '12px';
    this.node.style.display = 'flex';
    this.node.style.flexDirection = 'column';
    this.node.style.gap = '12px';
    this.node.style.height = '100%';
    this.node.style.backgroundColor = 'var(--jp-layout-color1)';

    // Title
    const title = document.createElement('h3');
    title.textContent = 'Context Chat';
    title.style.margin = '0';
    title.style.padding = '8px 0';
    title.style.borderBottom = '2px solid var(--jp-border-color1)';
    title.style.color = 'var(--jp-ui-font-color1)';

    // Chat log (conversation history)
    this._log = document.createElement('div');
    this._log.style.flex = '1';
    this._log.style.overflow = 'auto';
    this._log.style.border = '1px solid var(--jp-border-color2)';
    this._log.style.borderRadius = '4px';
    this._log.style.padding = '12px';
    this._log.style.backgroundColor = 'var(--jp-layout-color2)';
    this._log.style.fontFamily = 'var(--jp-code-font-family)';
    this._log.style.fontSize = '13px';
    this._log.style.lineHeight = '1.6';
    this._log.style.whiteSpace = 'pre-wrap';
    this._log.style.wordWrap = 'break-word';

    // Context toggle checkbox
    const contextContainer = document.createElement('div');
    contextContainer.style.display = 'flex';
    contextContainer.style.alignItems = 'center';
    contextContainer.style.gap = '8px';
    contextContainer.style.fontSize = '12px';
    contextContainer.style.color = 'var(--jp-ui-font-color2)';

    this._contextToggle = document.createElement('input');
    this._contextToggle.type = 'checkbox';
    this._contextToggle.checked = true;
    this._contextToggle.id = 'context-toggle';

    this._contextLabel = document.createElement('label');
    this._contextLabel.htmlFor = 'context-toggle';
    this._contextLabel.textContent = 'Include notebook context';
    this._contextLabel.style.cursor = 'pointer';

    contextContainer.appendChild(this._contextToggle);
    contextContainer.appendChild(this._contextLabel);

    // Input textarea
    this._input = document.createElement('textarea');
    this._input.placeholder = 'Ask about your notebook...';
    this._input.style.width = '100%';
    this._input.style.height = '80px';
    this._input.style.padding = '8px';
    this._input.style.border = '1px solid var(--jp-border-color2)';
    this._input.style.borderRadius = '4px';
    this._input.style.fontFamily = 'var(--jp-ui-font-family)';
    this._input.style.fontSize = '13px';
    this._input.style.resize = 'vertical';
    this._input.style.backgroundColor = 'var(--jp-layout-color1)';
    this._input.style.color = 'var(--jp-ui-font-color1)';

    // Allow Enter to send (Shift+Enter for new line)
    this._input.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._send.click();
      }
    });

    // Buttons container
    const buttonContainer = document.createElement('div');
    buttonContainer.style.display = 'flex';
    buttonContainer.style.gap = '8px';

    // Send button
    this._send = document.createElement('button');
    this._send.textContent = 'Send';
    this._send.style.flex = '1';
    this._send.style.padding = '10px';
    this._send.style.border = 'none';
    this._send.style.borderRadius = '4px';
    this._send.style.backgroundColor = 'var(--jp-brand-color1)';
    this._send.style.color = 'white';
    this._send.style.fontWeight = 'bold';
    this._send.style.cursor = 'pointer';
    this._send.style.fontSize = '13px';

    this._send.onmouseover = () => {
      this._send.style.backgroundColor = 'var(--jp-brand-color0)';
    };
    this._send.onmouseout = () => {
      this._send.style.backgroundColor = 'var(--jp-brand-color1)';
    };

    // Clear button
    this._clear = document.createElement('button');
    this._clear.textContent = 'Clear';
    this._clear.style.padding = '10px';
    this._clear.style.border = '1px solid var(--jp-border-color2)';
    this._clear.style.borderRadius = '4px';
    this._clear.style.backgroundColor = 'var(--jp-layout-color2)';
    this._clear.style.color = 'var(--jp-ui-font-color1)';
    this._clear.style.cursor = 'pointer';
    this._clear.style.fontSize = '13px';

    this._clear.onclick = () => {
      this.clearChat();
    };

    // Handle send
    this._send.onclick = async () => {
      const msg = this._input.value.trim();
      if (!msg) return;
      
      this._input.value = '';
      this.appendLine(`You: ${msg}`, 'user');

      this._send.disabled = true;
      this._send.textContent = 'Thinking...';

      try {
        const context = this._contextToggle.checked 
          ? getActiveNotebookContext(this._notebooks)
          : '';

        const data = await postJSON<{ reply: string }>(
          'context-chat/generate',
          { message: msg, context }
        );

        this.appendLine(`AI: ${data.reply}`, 'assistant');
      } catch (e: any) {
        this.appendLine(`Error: ${e?.message ?? String(e)}`, 'error');
      } finally {
        this._send.disabled = false;
        this._send.textContent = 'Send';
      }
    };

    // Assemble the UI
    buttonContainer.appendChild(this._send);
    buttonContainer.appendChild(this._clear);

    this.node.appendChild(title);
    this.node.appendChild(this._log);
    this.node.appendChild(contextContainer);
    this.node.appendChild(this._input);
    this.node.appendChild(buttonContainer);

    // Welcome message
    this.appendLine('Welcome to Context Chat!\n\nI can see your notebook and help with your code.\n\nTry asking me about your active cell or notebook.', 'system');
  }

  /**
   * Append a line to the chat log.
   */
  private appendLine(text: string, type: 'user' | 'assistant' | 'system' | 'error' = 'system') {
    const entry = document.createElement('div');
    entry.style.marginBottom = '12px';
    entry.style.padding = '8px';
    entry.style.borderRadius = '6px';
    entry.textContent = text;

    // Style by message type
    switch (type) {
      case 'user':
        entry.style.backgroundColor = 'var(--jp-brand-color3)';
        entry.style.borderLeft = '3px solid var(--jp-brand-color1)';
        break;
      case 'assistant':
        entry.style.backgroundColor = 'var(--jp-layout-color3)';
        entry.style.borderLeft = '3px solid var(--jp-success-color1)';
        break;
      case 'error':
        entry.style.backgroundColor = 'var(--jp-error-color3)';
        entry.style.borderLeft = '3px solid var(--jp-error-color1)';
        entry.style.color = 'var(--jp-error-color0)';
        break;
      default:
        entry.style.backgroundColor = 'var(--jp-layout-color3)';
        entry.style.fontStyle = 'italic';
        entry.style.opacity = '0.8';
    }

    this._log.appendChild(entry);
    this._log.scrollTop = this._log.scrollHeight;
  }

  /**
   * Clear the chat history.
   */
  public clearChat() {
    this._log.innerHTML = '';
    this.appendLine('Chat cleared. Start a new conversation!', 'system');
  }
}
