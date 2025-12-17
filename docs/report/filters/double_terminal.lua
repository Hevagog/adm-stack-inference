local WIDTH = 101


local function center_title(title, width)
  local padding = math.max(0, math.floor((width - #title) / 2))
  return string.rep(" ", padding) .. title
end

local function split_terminals(text, with_prefixes)
  local lines = {}
  for line in text:gmatch("([^\n]*)\n?") do
    table.insert(lines, line)
  end
  if #lines == 0 then
    return text
  end

  local t1_prefix = "T1: "
  local t2_prefix = "T2: "
  if not with_prefixes then
    t1_prefix = ""
    t2_prefix = ""
  end

  local shift = math.floor(WIDTH / 2)
  local pad = string.rep(" ", shift)
  local out = {}
  local in_block = false
  for _, line in ipairs(lines) do
    if not in_block then
      -- start of a block: '-> {' (arrow followed by optional spaces and '{')
      if line:match("^%s*%-%>%s*%{") then
        -- remove up to the opening brace
        local rest = line:gsub("^%s*%-%>%s*%{", "")
        if rest:match("%S") then
          table.insert(out, pad .. t2_prefix .. rest)
        end
        in_block = true
      elseif line:match("^%s*%-%>") then
        -- single-line arrow: remove arrow and optional following space, then shift
        local rest = line:gsub("^%s*%-%>%s?", "")
        if rest:match("%S") then
          table.insert(out, pad .. t2_prefix .. rest)
        end
      elseif line:match("%S") then
        table.insert(out, t1_prefix .. line)
      end
    else
      -- inside a '-> {' block: stop at a line that is exactly '}' (ignoring surrounding whitespace)
      if (line:gsub("^%s+", ""):gsub("%s+$", "")) == "}" then
        in_block = false
      else
        if line:match("%S") then
          table.insert(out, pad .. t2_prefix .. line)
        end
      end
    end
  end

  -- remove any remaining empty/whitespace-only lines and return
  local final = {}
  for _, ln in ipairs(out) do
    if ln:match("%S") then
      table.insert(final, ln)
    end
  end
  return table.concat(final, "\n")
end

function CodeBlock(el)
  if el.classes then
    local with_prefixes = true
    for _, cls in ipairs(el.classes) do
        if cls == "no-prefixes" then
            with_prefixes = false
            break
        end
    end
    for _, cls in ipairs(el.classes) do
        if cls == "double-terminal" then
            el.text = split_terminals(el.text, with_prefixes)
            return el
        end
    end
  end
  return el
end


-- Usage example:

-- ```python {.double-terminal .no-prefixes} -- .no-prefixes is optional
-- class CustomClass:
--     def __init__(self, name):
--         self.name = name

--     def greet(self):
--         return f"Hello, {self.name}!"

-- # Example usage
-- custom_obj = CustomClass("AI Enthusiast")
-- print(custom_obj.greet())

-- -> print("Data loaded successfully.")

-- -> {
--     for i in range(5):
--         print(f"Iteration {i}")
-- }
-- ```
