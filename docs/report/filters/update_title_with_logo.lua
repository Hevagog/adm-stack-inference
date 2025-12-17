-- update_title_with_logo.lua
local pandoc = require("pandoc")

local stringify = pandoc.utils and pandoc.utils.stringify

function Meta(meta)
    if meta["title-logo"] then
        -- Get string values for the logo source and existing title
        local logo_src = stringify(meta["title-logo"]) or ""
        local title_text = stringify(meta.title) or ""

        -- Determine width: allow raw string like '3cm' or numeric value (treated as cm)
        local width = "3cm"
        if meta["title-logo-width"] then
            if meta["title-logo-width"].t == "MetaString" then
                width = stringify(meta["title-logo-width"])
            elseif meta["title-logo-width"].t == "MetaInlines" then
                local num = tonumber(stringify(meta["title-logo-width"]))
                if num then
                    width = tostring(num) .. "cm"
                end
            end
        end

        -- Create an Image inline (empty caption, src set)
        local logo_image = pandoc.Image({}, logo_src)
        -- Attach width as an attribute so pandoc renderers can use it
        logo_image.attr = pandoc.Attr("", {}, {{"width", width}})

        -- Build new title as MetaInlines: logo, line break, original title
        local new_inlines = { logo_image, pandoc.LineBreak(), pandoc.Str(title_text) }
        meta.title = pandoc.MetaInlines(new_inlines)
    end

    return meta
end