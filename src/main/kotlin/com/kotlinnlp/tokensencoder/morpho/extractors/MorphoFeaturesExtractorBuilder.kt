/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho.extractors

import com.kotlinnlp.linguisticdescription.morphology.SingleMorphology
import com.kotlinnlp.linguisticdescription.morphology.morphologies.discourse.Punctuation
import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.*
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Article
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Noun
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Pronoun
import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Number as MorphoNumber

/**
 * The builder of [MorphoFeaturesExtractor].
 */
object MorphoFeaturesExtractorBuilder {

  /**
   * @param morphology a single morphology
   *
   * @return a morphological features extractor
   */
  operator fun invoke(morphology: SingleMorphology): MorphoFeaturesExtractor? = when (morphology) {
    is Verb -> VerbFeaturesExtractor(morphology)
    is Noun -> NounFeaturesExtractor(morphology)
    is Adjective -> AdjectiveFeaturesExtractor(morphology)
    is MorphoNumber -> NumberFeaturesExtractor(morphology)
    is Pronoun -> PronounFeaturesExtractor(morphology)
    is Article -> ArticleFeaturesExtractor(morphology)
    is Adverb -> AdverbFeaturesExtractor(morphology)
    is Preposition -> PrepositionFeaturesExtractor(morphology)
    is Conjunction -> ConjunctionFeaturesExtractor(morphology)
    is Punctuation -> PunctuationFeaturesExtractor(morphology)
    else -> GenericFeaturesExtractor(morphology)
  }
}
