/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.linguisticdescription.lexicon.LexiconDictionary
import com.kotlinnlp.morphologicalanalyzer.dictionary.MorphologyEntry
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalysis
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.neuralparser.language.Token
import com.kotlinnlp.tokensencoder.morpho.extractors.MorphoFeaturesExtractorBuilder
import com.kotlinnlp.neuraltokenizer.Token as TKToken

/**
 * The features extractor of the [MorphoEncoder].
 *
 * @param tokens the list of tokens
 * @param analyzer the morphological analyzer
 * @param langCode the language (iso-a2)
 * @param lexicalDictionary the lexical dictionary (can be null)
 */
class FeaturesExtractor(
  private val tokens: List<Token>,
  private val analyzer: MorphologicalAnalyzer,
  private val langCode: String,
  private val lexicalDictionary: LexiconDictionary?) {

  /**
   * The list of tokens.
   */
  private val tkTokens: List<TKToken> = this.tokens.toTKTokens()

  /**
   * The morphological analysis of the [tkTokens].
   */
  private val analysis: MorphologicalAnalysis = this.analyzer.analyze(
    text = this.tkTokens.joinToString { it.form },
    tokens = this.tkTokens,
    langCode = this.langCode)

  /**
   * @return a set of features for each token
   */
  fun extractFeatures(): List<Set<String>> {

    val tokensFeatures = mutableListOf<MutableSet<String>>()

    this.analysis.tokens.zip(this.tkTokens).filterNot { (_, token) -> token.isSpace }.forEach { (entries, token) ->

      val tokenFeaturesSet = mutableSetOf<String>()

      entries?.map { tokenFeaturesSet.addAll(it.toFeatures()) } ?: tokenFeaturesSet.add("i:0 _")

      tokensFeatures.add(tokenFeaturesSet)
    }

    this.addMWEFeatures(tokensFeatures)

    return tokensFeatures
  }

  /**
   * @param tokensFeatures a list of the same size of the [tokens]
   */
  private fun addMWEFeatures(tokensFeatures: List<MutableSet<String>>) {

    if (this.analysis.multiWords.isNotEmpty()) {

      this.analysis.multiWords.forEach { multiword ->

        (multiword.startToken .. multiword.endToken).forEach { tokenIndex ->

          val tokenFeaturesSet = mutableSetOf<String>()

          multiword.morphologies.map { tokenFeaturesSet.addAll(it.toFeatures()) }

          tokensFeatures[tokenIndex].addAll(tokenFeaturesSet)
        }
      }
    }
  }

  /**
   * Transform a [MorphologyEntry] in a list of features.
   *
   * @return a list of features
   */
  private fun MorphologyEntry.toFeatures(): List<String> {

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

  /**
   * @return a list of [TKToken]s originated from this [Token]s
   */
  private fun List<Token>.toTKTokens(): List<TKToken> = this.map {
    TKToken(id = it.id, form = it.word, startAt = 0, endAt = 0, isSpace = false)
  }
}