-- Apoio ao lmfquant.sty: chave de ordenação do índice remissivo sem acentos.
local mapa = {
  ["á"]="a", ["à"]="a", ["â"]="a", ["ã"]="a", ["é"]="e", ["ê"]="e", ["í"]="i",
  ["ó"]="o", ["ô"]="o", ["õ"]="o", ["ú"]="u", ["ü"]="u", ["ç"]="c",
  ["Á"]="a", ["À"]="a", ["Â"]="a", ["Ã"]="a", ["É"]="e", ["Ê"]="e", ["Í"]="i",
  ["Ó"]="o", ["Ô"]="o", ["Õ"]="o", ["Ú"]="u", ["Ç"]="c", ["²"]="2",
}

function lmfchave(texto)
  local k = texto:gsub("\\%a+%s*", ""):gsub("[{}$^_\\]", "")
  k = k:gsub(utf8.charpattern, function(c) return mapa[c] or c end)
  tex.sprint(-2, k:lower())
end

-- Normaliza uma entrada de \index: cada nível sem chave explícita
-- ("texto") vira "chave sem acento@texto", para que entradas criadas
-- por \termo e por \index caiam na mesma linha do índice.
function lmfindice(texto)
  local niveis = {}
  for nivel in (texto .. "!"):gmatch("(.-)!") do
    local arroba = nivel:find("@", 1, true)
    if arroba then                       -- chave explícita: só uniformiza
      local k = nivel:sub(1, arroba - 1)
      k = k:gsub(utf8.charpattern, function(c) return mapa[c] or c end)
      nivel = k:lower() .. nivel:sub(arroba)
    elseif not nivel:find("|", 1, true) then
      local k = nivel:gsub("\\%a+%s*", ""):gsub("[{}$^_\\]", "")
      k = k:gsub(utf8.charpattern, function(c) return mapa[c] or c end)
      nivel = k:lower() .. "@" .. nivel
    end
    niveis[#niveis + 1] = nivel
  end
  tex.sprint(-2, table.concat(niveis, "!"))
end
