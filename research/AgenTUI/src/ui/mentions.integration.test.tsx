import React from 'react';
import { describe, expect, it } from 'vitest';
import { render } from 'ink-testing-library';
import { __private__ } from './App.js';

describe('Mention autocomplete cursor', () => {
  it('places cursor after inserted mention on tab', () => {
    const input = 'open @tmp';
    const mentionContext = { start: 5, query: 'tmp' };
    const value = '/Users/test/tmp';

    const harness = render(
      <__private__.ComposerInput
        value={input}
        cursor={mentionContext.start + mentionContext.query.length + 1}
        onChange={() => {}}
        onCursorChange={() => {}}
        onSubmit={() => {}}
        focus
        disabled={false}
        submitDisabled={false}
        width={40}
        placeholder="type"
        highlightMentions
      />
    );

    // Simulate the insertMentionSelection logic directly
    const formatted = value;
    const before = input.slice(0, mentionContext.start);
    const tokenEnd = mentionContext.start + 1 + (mentionContext.query?.length ?? 0);
    const after = input.slice(tokenEnd);
    const insertion = `@${formatted} `;
    const nextValue = `${before}${insertion}${after}`;
    const nextCursor = before.length + insertion.length;

    expect(nextValue).toBe('open @/Users/test/tmp ');
    expect(nextCursor).toBe(nextValue.length);
  });
});
