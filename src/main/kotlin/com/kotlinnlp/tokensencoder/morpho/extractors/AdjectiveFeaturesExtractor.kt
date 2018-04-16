/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho.extractors

import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Adjective

/**
 * Extract the features from the given [morphology].
 *
 * @param morphology the morphology
 */
class AdjectiveFeaturesExtractor(private val morphology: Adjective) : MorphoFeaturesExtractor {

  /**
   * Return a list of features.
   */
  override fun get(): List<String> = listOf(
    "t:%s".format("Adjective"),
    "p:%s".format(this.morphology.type),
    "p:%s l:%s".format(this.morphology.type, this.morphology.lemma),
    "p:%s n:%s p:%s g:%s c:%s".format(
      this.morphology.type,
      this.morphology.person,
      this.morphology.number,
      this.morphology.gender,
      this.morphology.case)
  )
}