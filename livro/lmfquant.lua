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
