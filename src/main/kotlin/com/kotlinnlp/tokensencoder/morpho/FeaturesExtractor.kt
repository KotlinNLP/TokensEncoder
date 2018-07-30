/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.linguisticdescription.lexicon.LexiconDictionary
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.sentence.MorphoSentence
import com.kotlinnlp.tokensencoder.morpho.extractors.MorphoFeaturesExtractorBuilder

/**
 * The features extractor of the [MorphoEncoder].
 *
 * @param sentence the sentence
 * @param lexicalDictionary the lexical dictionary (can be null)
 */
class FeaturesExtractor(
  private val sentence: MorphoSentence,
  private val lexicalDictionary: LexiconDictionary?
) {

  /**
   * @return a set of features for each token
   */
  fun extractFeatures(): List<Set<String>> = this.sentence.tokens.map { token ->

    val tokenFeaturesSet = mutableSetOf<String>()

    token.morphologies.forEach {
      tokenFeaturesSet.addAll(it.toFeatures())
    }

    if (tokenFeaturesSet.isEmpty()) tokenFeaturesSet.add("i:0 p:unknown")

    tokenFeaturesSet
  }

  /**
   * Transform a [Morphology] in a list of features.
   *
   * @return a list of features
   */
  private fun Morphology.toFeatures(): List<String> {

    val list = mutableListOf<String>()

    this.list.forEachIndexed { index, morphology ->

      list.addAll(morphology.getMorphoFeatures().map { "i:$index $it" })
      list.addAll(morphology.getLexicalFeatures().map { "i:$index $it" })
    }

    return list
  }

  /**
   * @return a list o morphological features
   */
  private fun SingleMorphology.getMorphoFeatures(): List<String> =
    MorphoFeaturesExtractorBuilder(this).get()

  /**
   * @return a list of lexical features
   */
  private fun SingleMorphology.getLexicalFeatures(): List<String> {

    val list = mutableListOf<String>()

    this@FeaturesExtractor.lexicalDictionary?.get(
      lemma = this.lemma,
      posTag = this.type.baseAnnotation)?.syntax?.let { syntacticInfo ->

      syntacticInfo.regencies?.let {
        list.addAll(it.map { regency ->
          "p:%s r:%s".format(this.type, regency) })
      }

      syntacticInfo.subcategorization?.let {
        list.addAll(it.map { subcategory ->
          "p:%s s:%s".format(this.type, subcategory) })
      }
    }

    return list
  }
}
