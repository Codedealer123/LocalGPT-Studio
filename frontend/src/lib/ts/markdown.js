import katex from 'katex';
import 'katex/dist/katex.min.css';

function escapeHtml(value = "") {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function sanitizeUrl(url = "") {
  const value = String(url).trim();
  return /^(https?:|mailto:|#|\/)/i.test(value) ? value : '#';
}

// Block LaTeX environments - rendered as display math.
const BLOCK_ENVS = new Set([
  'equation', 'equation*',
  'align', 'align*',
  'alignat', 'alignat*',
  'gather', 'gather*',
  'multline', 'multline*',
  'flalign', 'flalign*',
  'split', 'array',
  'matrix', 'pmatrix', 'bmatrix', 'Bmatrix', 'vmatrix', 'Vmatrix',
  'cases', 'eqnarray', 'eqnarray*',
]);

// Map numbered environments to their starred (unnumbered) equivalents so
// KaTeX does not render "(1)", "(2)" tags in the output.
const NUMBERED_ENVS = {
  'equation': 'equation*',
  'align':    'align*',
  'alignat':  'alignat*',
  'gather':   'gather*',
  'multline': 'multline*',
  'flalign':  'flalign*',
  'eqnarray': 'eqnarray*',
};

function extractMathTokens(text) {
  const tokens = [];

  // \begin{env}...\end{env} environments
  text = text.replace(/\\begin\{([^}]+)\}([\s\S]*?)\\end\{\1\}/g, (_, env, body) => {
    const envName = env.trim();
    const isBlock = BLOCK_ENVS.has(envName);
    const renderEnv = NUMBERED_ENVS[envName] ?? envName;
    const token = `__MATH_${isBlock ? 'BLOCK' : 'INLINE'}_${tokens.length}__`;
    tokens.push({ type: isBlock ? 'block' : 'inline', content: `\\begin{${renderEnv}}${body}\\end{${renderEnv}}` });
    return token;
  });

  // Block math: \[...\] and $$...$$
  text = text.replace(/\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g, (_, a, b) => {
    const token = `__MATH_BLOCK_${tokens.length}__`;
    tokens.push({ type: 'block', content: (a ?? b).trim() });
    return token;
  });

  // Inline math: \(...\) and $...$
  text = text.replace(/\\\(([\s\S]*?)\\\)|\$([^\$\n]+?)\$/g, (_, a, b) => {
    const token = `__MATH_INLINE_${tokens.length}__`;
    tokens.push({ type: 'inline', content: (a ?? b).trim() });
    return token;
  });

  return { text, tokens };
}

function renderMathTokens(html, tokens) {
  return html.replace(/__MATH_(BLOCK|INLINE)_(\d+)__/g, (_, type, i) => {
    const { content, type: kind } = tokens[Number(i)];
    try {
      return katex.renderToString(content, {
        displayMode: kind === 'block',
        throwOnError: false,
        output: 'html',
      });
    } catch {
      return escapeHtml(content);
    }
  });
}

function parseInline(text = "", tokens = []) {
  const codeTokens = [];

  // Protect math placeholders FIRST. The bold regex /(\*\*|__)(.*?)\1/ would
  // treat the leading/trailing __ of __MATH_BLOCK_0__ as bold delimiters and
  // destroy the placeholder. Swap to a sentinel with no underscores or asterisks.
  const mathPlaceholders = [];
  text = text.replace(/__MATH_(BLOCK|INLINE)_(\d+)__/g, (m) => {
    const id = `XMATHX${mathPlaceholders.length}XMATHX`;
    mathPlaceholders.push(m);
    return id;
  });

  let html = escapeHtml(text);

  html = html.replace(/`([^`]+)`/g, (_, code) => {
    const token = `__CODE_${codeTokens.length}__`;
    codeTokens.push(`<code>${escapeHtml(code)}</code>`);
    return token;
  });

  html = html.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_, alt, src) => {
    return `<img src="${sanitizeUrl(src)}" alt="${escapeHtml(alt)}" />`;
  });

  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => {
    return `<a href="${sanitizeUrl(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`;
  });

  html = html.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/(\*\*|__)(.*?)\1/g, '<strong>$2</strong>');
  html = html.replace(/(\*|_)(.*?)\1/g, '<em>$2</em>');
  html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');
  html = html.replace(/==(.+?)==/g, '<mark>$1</mark>');
  html = html.replace(/\^([^^\n]+)\^/g, '<sup>$1</sup>');
  html = html.replace(/~([^~\n]+)~/g, '<sub>$1</sub>');
  html = html.replace(/ {2}\n/g, '<br />');
  html = html.replace(/\n/g, '<br />');

  html = codeTokens.reduce(
    (result, token, index) => result.replace(`__CODE_${index}__`, token),
    html
  );

  // Restore math placeholders so the caller's renderMathTokens pass can find them.
  html = mathPlaceholders.reduce(
    (result, original, i) => result.replace(`XMATHX${i}XMATHX`, original),
    html
  );

  return html;
}

function getListItemMeta(line = "") {
  const match = line.match(/^(\s*)(?:([-*+])\s+(?:\[( |x|X)\]\s+)?|(\d+)\.\s+)(.*)$/);
  if (!match) return null;

  const indent = match[1]?.length ?? 0;
  const checkbox = match[3];
  const orderedStart = match[4];
  const content = match[5] ?? '';

  return {
    indent,
    checkbox,
    orderedStart,
    content,
    kind: checkbox !== undefined ? 'task' : orderedStart ? 'ordered' : 'unordered',
  };
}

function parseList(lines, startIndex, tokens) {
  const firstItem = getListItemMeta(lines[startIndex]);
  if (!firstItem) return null;

  const items = [];
  let index = startIndex;
  const listKind = firstItem.kind;
  const baseIndent = firstItem.indent;
  const start = firstItem.orderedStart ? Number(firstItem.orderedStart) : 1;

  while (index < lines.length) {
    const item = getListItemMeta(lines[index]);
    if (!item || item.indent !== baseIndent || item.kind !== listKind) break;

    if (item.kind === 'task') {
      items.push(
        `<li class="task-list-item"><input type="checkbox" disabled ${/x/i.test(item.checkbox) ? 'checked' : ''} /> <span>${parseInline(item.content, tokens)}</span></li>`
      );
    } else {
      items.push(`<li>${parseInline(item.content, tokens)}</li>`);
    }

    index += 1;

    while (index < lines.length) {
      const continuation = lines[index];
      if (!continuation.trim()) break;

      const continuationItem = getListItemMeta(continuation);
      const continuationIndent = continuation.match(/^(\s*)/)?.[1]?.length ?? 0;

      if (continuationItem || continuationIndent <= baseIndent) break;

      const previous = items.pop() ?? '';
      items.push(previous.replace('</li>', `<br />${parseInline(continuation.trim(), tokens)}</li>`));
      index += 1;
    }
  }

  const tag = listKind === 'ordered' ? 'ol' : 'ul';
  const startAttr = tag === 'ol' && start > 1 ? ` start="${start}"` : '';
  const classAttr = listKind === 'task' ? ' class="task-list"' : '';

  return {
    nextIndex: index,
    html: `<${tag}${startAttr}${classAttr}>${items.join('')}</${tag}>`
  };
}

function parseTable(lines, startIndex, tokens) {
  if (startIndex + 1 >= lines.length) return null;
  if (!/\|/.test(lines[startIndex]) || !/^\s*\|?[\s:-|]+\|?\s*$/.test(lines[startIndex + 1])) {
    return null;
  }

  const rows = [];
  let index = startIndex;

  while (index < lines.length && /\|/.test(lines[index]) && lines[index].trim() !== '') {
    rows.push(lines[index]);
    index += 1;
  }

  const splitRow = (line) =>
    line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim());

  const header = splitRow(rows[0]);
  const body = rows.slice(2).map(splitRow);

  return {
    nextIndex: index,
    html:
      `<table><thead><tr>${header.map((cell) => `<th>${parseInline(cell, tokens)}</th>`).join('')}</tr></thead>` +
      `<tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${parseInline(cell, tokens)}</td>`).join('')}</tr>`).join('')}</tbody></table>`
  };
}

function extractThinkBlock(markdown) {
  const match = markdown.match(/^<think>([\s\S]*?)<\/think>\n*/);
  if (!match) return { think: null, rest: markdown };
  return {
    think: match[1].trim(),
    rest: markdown.slice(match[0].length),
  };
}

// Lightweight inline-only renderer for user messages.
// No block elements (no <p>, <div>, <ul> etc.) so the bubble never grows.
// Uses only * for italic (not _) to avoid breaking snake_case identifiers.
export function userMarkdownToHtml(text = "") {
  let html = escapeHtml(String(text));
  // inline code — protect first so inner content isn't touched by other regexes
  const codeTokens = [];
  html = html.replace(/`([^`]+)`/g, (_, code) => {
    const token = `__UCODE_${codeTokens.length}__`;
    codeTokens.push(`<code>${escapeHtml(code)}</code>`);
    return token;
  });
  html = html.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');
  html = html.replace(/\n/g, '<br />');
  html = codeTokens.reduce((r, tok, i) => r.replace(`__UCODE_${i}__`, tok), html);
  return html;
}

export function markdownToHtml(markdown = "", appendHtml = "") {
  const { think, rest } = extractThinkBlock(String(markdown).replace(/\r\n/g, '\n'));
  let source = rest;

  const extracted = extractMathTokens(source);
  source = extracted.text;
  const mathTokens = extracted.tokens;

  const lines = source.split('\n');
  const blocks = [];

  for (let index = 0; index < lines.length;) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      index += 1;
      continue;
    }

    if (/^__MATH_BLOCK_\d+__$/.test(trimmed)) {
      blocks.push(`<div class="math-block">${renderMathTokens(trimmed, mathTokens)}</div>`);
      index += 1;
      continue;
    }

    if (/^```/.test(trimmed)) {
      const language = trimmed.slice(3).trim();
      const codeLines = [];
      index += 1;

      while (index < lines.length && !/^```/.test(lines[index].trim())) {
        codeLines.push(lines[index]);
        index += 1;
      }

      if (index < lines.length) index += 1;
      blocks.push(`<pre><code${language ? ` class="language-${escapeHtml(language)}"` : ''}>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
      continue;
    }

    const table = parseTable(lines, index, mathTokens);
    if (table) {
      blocks.push(table.html);
      index = table.nextIndex;
      continue;
    }

    if (/^\s*(?:[-*+]\s+(?:\[[ xX]\]\s+)?|\d+\.\s+)/.test(line)) {
      const list = parseList(lines, index, mathTokens);
      if (!list) { index += 1; continue; }
      blocks.push(list.html);
      index = list.nextIndex;
      continue;
    }

    if (/^>\s?/.test(trimmed)) {
      const quoteLines = [];
      while (index < lines.length && /^>\s?/.test(lines[index].trim())) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ''));
        index += 1;
      }
      blocks.push(`<blockquote>${quoteLines.map((q) => `<p>${parseInline(q, mathTokens)}</p>`).join('')}</blockquote>`);
      continue;
    }

    if (/^#{1,6}\s+/.test(trimmed)) {
      const [, hashes, content] = trimmed.match(/^(#{1,6})\s+(.*)$/);
      const level = hashes.length;
      blocks.push(`<h${level}>${parseInline(content, mathTokens)}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      blocks.push('<hr />');
      index += 1;
      continue;
    }

    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^```/.test(lines[index].trim()) &&
      !/^>\s?/.test(lines[index].trim()) &&
      !/^#{1,6}\s+/.test(lines[index].trim()) &&
      !/^\s*(?:[-*+]\s+(?:\[[ xX]\]\s+)?|\d+\.\s+)/.test(lines[index]) &&
      !(/\|/.test(lines[index]) && index + 1 < lines.length && /^\s*\|?[\s:-|]+\|?\s*$/.test(lines[index + 1]))
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }

    blocks.push(`<p>${parseInline(paragraph.join('\n'), mathTokens)}</p>`);
  }

  // Single final pass renders all math tokens across every block type.
  const rendered = blocks.map((b) => renderMathTokens(b, mathTokens));

  if (!appendHtml) {
    const thinkHtml = think
      ? `<details class="think-block"><summary>Thinking</summary><div class="think-body">${markdownToHtml(think)}</div></details>`
      : '';
    return thinkHtml + rendered.join('');
  }

  if (rendered.length === 0) return `<p>${appendHtml}</p>`;

  const last = rendered[rendered.length - 1];
  const closing = last.match(/<\/([a-z][a-z0-9]*)>$/i);
  if (closing) {
    rendered[rendered.length - 1] = last.slice(0, -closing[0].length) + appendHtml + closing[0];
  } else {
    rendered[rendered.length - 1] = last + appendHtml;
  }

  const thinkHtml = think
    ? `<details class="think-block"><summary>Thinking</summary><div class="think-body">${markdownToHtml(think)}</div></details>`
    : '';

  return thinkHtml + rendered.join('');
}