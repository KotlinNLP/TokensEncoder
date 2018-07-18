/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.linguisticdescription.lexicon.LexiconDictionary
import com.kotlinnlp.linguisticdescription.morphology.Morphology
import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.MorphoToken
import com.kotlinnlp.tokensencoder.morpho.extractors.MorphoFeaturesExtractorBuilder

/**
 * The features extractor of the [MorphoEncoder].
 *
 * @param sentence the sentence
 * @param lexicalDictionary the lexical dictionary (can be null)
 */
class FeaturesExtractor(
  private val sentence: Sentence<MorphoToken>,
  private val lexicalDictionary: LexiconDictionary?) {

  /**
   * @return a set of features for each token
   */
  fun extractFeatures(): List<Set<String>> {

    val tokensFeatures = mutableListOf<MutableSet<String>>()

    this.sentence.tokens.forEach { token ->

      val tokenFeaturesSet = mutableSetOf<String>()

      token.lexicalForms.forEach {
        tokenFeaturesSet.addAll(it.toFeatures())
      }

      /*
        TODO: handle multi-words
        this.sentence.getInvolvedMultiWords(tokenIndex)?.forEach {
          it.morphologies.map { morphology -> tokenFeaturesSet.addAll(morphology.toFeatures()) }
        }
      */

      if (tokenFeaturesSet.isEmpty()) tokenFeaturesSet.add("i:0 _")

      tokensFeatures.add(tokenFeaturesSet)
    }

    return tokensFeatures
  }

  /**
   * Transform a [Morphology] in a list of features.
   *
   * @return a list of features
   */
  private fun Morphology.toFeatures(): List<String> {

    val list = mutableListOf<String>()

    this.list.forEachIndexed { index, morphology ->

      list.addAll(

        MorphoFeaturesExtractorBuilder(morphology)
          ?.get()
          ?.map { "i:$index $it" }
          ?: listOf("i:%d p:%s".format(index, morphology.type))
      )

      this@FeaturesExtractor.lexicalDictionary?.get(
        lemma = morphology.lemma,
        posTag = morphology.type.baseAnnotation)?.syntax?.let { syntacticInfo ->

        syntacticInfo.regencies?.let {
          list.addAll(it.map { "i:%d p:%s r:%s".format(index, morphology.type, it) })
        }

        syntacticInfo.subcategorization?.let {
          list.addAll(it.map { "i:%d p:%s s:%s".format(index, morphology.type, it) })
        }
      }
    }

    return list
  }
}
