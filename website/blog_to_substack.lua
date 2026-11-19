-- Pandoc Lua filter used by `website/blog_to_substack.py`.
--
-- Adapts a blog post for the Substack editor:
-- - Removes HTML comments and `<div>` wrappers (raw HTML Substack ignores), keeping
--   the text inside them, e.g., figure captions
-- - Shows links whose text is a full URL without `https://`: in Google Docs such
--   links become "smart chips" that make Substack drop the whole paste
-- - Splits a list at each code block: Substack moves code blocks out of list
--   items, which leaves the following sub-bullets under an empty bullet
-- - With `tables=code`, renders each table as aligned text in a code block,
--   since Substack has no tables

local tables_mode = 'code'

function Meta(meta)
  if meta.substack_tables then
    tables_mode = pandoc.utils.stringify(meta.substack_tables)
  end
end

function RawBlock(el)
  if el.format:match('html') then
    local t = el.text
    if t:match('^%s*<!%-%-') then
      return {}
    end
    if t:match('^%s*</?div') then
      -- Drop the `<div>` tags but keep any text inside, e.g., figure captions.
      local inner = t:gsub('</?div[^>]*>', '')
      if inner:match('%S') then
        return pandoc.read(inner, 'gfm-tex_math_dollars').blocks
      end
      return {}
    end
  end
end

function RawInline(el)
  if el.format:match('html') and el.text:match('^%s*<!%-%-') then
    return {}
  end
end

function Div(el)
  return el.content
end

function Link(el)
  local txt = pandoc.utils.stringify(el.content)
  if txt:match('^https?://') then
    local short = txt:gsub('^https?://', ''):gsub('^www%.', ''):gsub('/$', '')
    el.content = { pandoc.Str(short) }
    return el
  end
end

function Table(el)
  if tables_mode == 'code' then
    local txt = pandoc.write(pandoc.Pandoc({ el }), 'plain', pandoc.WriterOptions({ columns = 200 }))
    return pandoc.CodeBlock((txt:gsub('%s+$', '')))
  end
end

-- Split `lst` at every item that contains a code block at its top level.
local function split_list(lst)
  local has_code = false
  for _, item in ipairs(lst.content) do
    for _, b in ipairs(item) do
      if b.t == 'CodeBlock' then has_code = true end
    end
  end
  if not has_code then return { lst } end
  local out, cur = {}, {}
  local is_ordered = lst.t == 'OrderedList'
  local start = is_ordered and lst.listAttributes.start or 1
  local num = start
  local cur_start = start
  local function flush()
    if #cur > 0 then
      if is_ordered then
        local attrs = pandoc.ListAttributes(cur_start, lst.listAttributes.style, lst.listAttributes.delimiter)
        table.insert(out, pandoc.OrderedList(cur, attrs))
      else
        table.insert(out, pandoc.BulletList(cur))
      end
    end
    cur = {}
  end
  for _, item in ipairs(lst.content) do
    local head, tail, seen = {}, {}, false
    for _, b in ipairs(item) do
      if b.t == 'CodeBlock' then seen = true end
      if seen then table.insert(tail, b) else table.insert(head, b) end
    end
    if #cur == 0 then cur_start = num end
    num = num + 1
    if not seen then
      table.insert(cur, item)
    else
      table.insert(cur, head)
      flush()
      cur_start = num
      for _, b in ipairs(tail) do
        if b.t == 'BulletList' or b.t == 'OrderedList' then
          for _, x in ipairs(split_list(b)) do table.insert(out, x) end
        else
          table.insert(out, b)
        end
      end
    end
  end
  flush()
  return out
end

function Blocks(blocks)
  local out = {}
  for _, b in ipairs(blocks) do
    if b.t == 'BulletList' or b.t == 'OrderedList' then
      for _, x in ipairs(split_list(b)) do table.insert(out, x) end
    else
      table.insert(out, b)
    end
  end
  return out
end

return {
  { Meta = Meta },
  { RawBlock = RawBlock, RawInline = RawInline, Div = Div, Link = Link, Table = Table },
  { Blocks = Blocks },
}
