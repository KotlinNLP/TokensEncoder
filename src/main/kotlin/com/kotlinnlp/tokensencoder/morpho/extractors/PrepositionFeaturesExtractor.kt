/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho.extractors

import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Preposition

/**
 * Extract the features from the given [morphology].
 *
 * @param morphology the morphology
 */
class PrepositionFeaturesExtractor(private val morphology: Preposition) : MorphoFeaturesExtractor {

  /**
   * Return a list of features.
   */
  override fun get(): List<String> = listOf(
    "t:%s".format("Preposition"),
    "p:%s".format(this.morphology.type),
    "p:%s l:%s".format(this.morphology.type, this.morphology.lemma)
  )
}