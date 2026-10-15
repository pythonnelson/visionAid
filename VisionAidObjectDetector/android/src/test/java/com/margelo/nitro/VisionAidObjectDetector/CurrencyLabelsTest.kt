package com.margelo.nitro.VisionAidObjectDetector

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CurrencyLabelsTest {

    private val master = File("../../ml/currency/labels.json")
    private val asset = File("src/main/assets/currency_labels.json")

    @Test fun masterLabelsParseAndMapIndicesToDenominations() {
        val l = CurrencyLabels.fromJson(master.readText())
        assertEquals(10, l.size)
        assertEquals("New Leone, one", l.classAt(0)!!.spokenEn)
        assertEquals("New Leone, five", l.classAt(2)!!.spokenEn)
        assertEquals("New Leone, twenty", l.classAt(4)!!.spokenEn)
        assertEquals("Old Leone, one thousand", l.classAt(5)!!.spokenEn)
        assertEquals("Old Leone, five thousand", l.classAt(7)!!.spokenEn)
        assertEquals("Old Leone, ten thousand", l.classAt(8)!!.spokenEn)
        assertEquals("NO_NOTE", l.classAt(9)!!.id)
        assertFalse(l.classAt(9)!!.isNote)
        assertNull(l.classAt(10)); assertNull(l.classAt(-1))
    }

    @Test fun oldAndNewSeriesAreNeverMerged() {
        val l = CurrencyLabels.fromJson(master.readText())
        val ids = l.classes.map { it.id }
        assertTrue("NEW_1" in ids && "OLD_1000" in ids)
        assertEquals(setOf("new"), l.classes.filter { it.id.startsWith("NEW_") }.map { it.series }.toSet())
        assertEquals(setOf("old"), l.classes.filter { it.id.startsWith("OLD_") }.map { it.series }.toSet())
        assertTrue(l.classes.filter { it.isNote }.all { it.spokenEn.startsWith(if (it.series == "new") "New Leone" else "Old Leone") })
        assertFalse("there must be no old Le 20 class", ids.contains("OLD_20"))
    }

    @Test fun androidAssetCopyIsIdenticalToMasterLabels() {
        assertTrue("run export_tflite.py", asset.exists())
        assertEquals(master.readText(), asset.readText())
    }

    private fun json(vararg rows: String) = """{"classes":[${rows.joinToString(",")}]}"""
    private fun row(i: Int, id: String, series: String = "new", spoken: String = "x") =
        """{"index":$i,"id":"$id","series":"$series","denomination":1,"spoken_en":"$spoken"}"""

    @Test(expected = IllegalArgumentException::class) fun rejectsWrongIndexOrder() {
        CurrencyLabels.fromJson(json(row(1, "NEW_1"), row(0, "NO_NOTE", "none", "")))
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsMissingNoNote() {
        CurrencyLabels.fromJson(json(row(0, "NEW_1"), row(1, "NEW_2")))
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsNoteWithoutSpokenLabel() {
        CurrencyLabels.fromJson(json(row(0, "NEW_1", spoken = ""), row(1, "NO_NOTE", "none", "")))
    }

    @Test(expected = IllegalArgumentException::class) fun rejectsDuplicateIds() {
        CurrencyLabels.fromJson(json(row(0, "NEW_1"), row(1, "NEW_1"), row(2, "NO_NOTE", "none", "")))
    }
}
