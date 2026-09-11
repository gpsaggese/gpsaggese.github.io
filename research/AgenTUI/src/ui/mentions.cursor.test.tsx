import { describe, expect, it } from 'vitest';
import { applyMentionInsertion } from './mentions.js';
import { renderComposerView } from './composer-renderer.js';

describe('Composer mention autocomplete cursor integration', () => {
  it('places cursor at the end after mention insertion plus extra text', () => {
    const base = '';
    const ctx = { start: 0, query: '' };
    const inserted = applyMentionInsertion(base, ctx, '/Users/test/tmp/fs-spec');
    const value = `${inserted.nextValue}more`;
    const cursor = value.length;

    const view = renderComposerView(value, cursor, 80);
    const lastLine = view.displayLines[view.displayLines.length - 1];

    expect(view.cursorRow).toBe(view.displayLines.length - 1);
    expect(view.cursorColumn).toBe(lastLine.columns[lastLine.columns.length - 1]);
  });
});
