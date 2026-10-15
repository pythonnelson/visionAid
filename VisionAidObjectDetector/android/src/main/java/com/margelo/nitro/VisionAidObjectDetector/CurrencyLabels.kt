package com.margelo.nitro.VisionAidObjectDetector

import org.json.JSONObject

/** One banknote class as described in labels.json (the single source of truth for class order). */
data class CurrencyClass(
    val index: Int,
    val id: String,
    val series: String,        // "old", "new" or "none"
    val denomination: Int,
    val spokenEn: String
) {
    val isNote: Boolean get() = series == "old" || series == "new"
}

/**
 * Class index -> denomination mapping for the currency model.
 * Loaded from `currency_labels.json` (copied from ml/currency/labels.json by the export script).
 * Validation is strict: a wrong or reordered file must fail loudly instead of announcing a wrong amount.
 */
class CurrencyLabels(val classes: List<CurrencyClass>) {

    init {
        require(classes.isNotEmpty()) { "currency labels are empty" }
        classes.forEachIndexed { i, c ->
            require(c.index == i) { "class '${c.id}' has index ${c.index} but is at position $i" }
            if (c.isNote) require(c.spokenEn.isNotBlank()) { "note class '${c.id}' has no spoken label" }
        }
        require(classes.map { it.id }.toSet().size == classes.size) { "duplicate class ids" }
        require(noNoteIndex >= 0) { "labels must contain a NO_NOTE class" }
        require(classes.last().id == "NO_NOTE") { "NO_NOTE must be the last class (as trained)" }
    }

    val size: Int get() = classes.size
    val noNoteIndex: Int get() = classes.indexOfFirst { it.id == "NO_NOTE" }

    fun classAt(index: Int): CurrencyClass? = classes.getOrNull(index)

    companion object {
        fun fromJson(json: String): CurrencyLabels {
            val arr = JSONObject(json).getJSONArray("classes")
            val list = (0 until arr.length()).map {
                val o = arr.getJSONObject(it)
                CurrencyClass(o.getInt("index"), o.getString("id"), o.getString("series"),
                    o.getInt("denomination"), o.optString("spoken_en", ""))
            }
            return CurrencyLabels(list)
        }
    }
}
