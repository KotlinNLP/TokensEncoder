/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho

import com.kotlinnlp.linguisticdescription.lexicon.LexiconDictionary
import com.kotlinnlp.linguisticdescription.morphology.dictionary.MorphologyDictionary
import com.kotlinnlp.morphologicalanalyzer.MorphologicalAnalyzer
import com.kotlinnlp.neuralparser.language.Sentence
import com.kotlinnlp.progressindicator.ProgressIndicatorBar
import com.kotlinnlp.simplednn.utils.DictionarySet

/**
 * Collect all the possible features from the given given [sentences].
 *
 * @param dictionary the morphological dictionary
 * @param lexicalDictionary the lexicon dictionary (can be null)
 * @param langCode the language code
 * @param sentences the list of sentences
 */
class FeaturesCollector(
  private val dictionary: MorphologyDictionary,
  private val lexicalDictionary: LexiconDictionary?,
  private val langCode: String,
  private val sentences: List<Sentence>
) {

  /**
   * @return the set of features collected from the sentences
   */
  fun collect(): DictionarySet<String> {

    val featuresDictionary = DictionarySet<String>()
    val progress = ProgressIndicatorBar(total = sentences.size)

    this.sentences.forEach { sentence ->

      progress.tick()

      val tokenFeatures: List<Set<String>> = FeaturesExtractor(
        tokens = sentence.tokens,
        analyzer = MorphologicalAnalyzer(this.dictionary),
        langCode = this.langCode,
        lexicalDictionary = lexicalDictionary).extractFeatures()

      tokenFeatures.forEach { featuresDictionary.addAll(it) }
    }

    return featuresDictionary
  }

  /**
   * Add the given [features] to the dictionary set.
   *
   * @param features a set of features
   */
  private fun DictionarySet<String>.addAll(features: Set<String>) {
    features.forEach { this.add(it) }
  }
}