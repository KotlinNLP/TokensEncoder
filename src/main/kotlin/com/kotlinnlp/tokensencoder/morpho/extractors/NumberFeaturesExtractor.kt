/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho.extractors

import com.kotlinnlp.linguisticdescription.morphology.morphologies.things.Number as MorphoNumber

/**
 * Extract the features from the given [morphology].
 *
 * @param morphology the morphology
 */
class NumberFeaturesExtractor(private val morphology: MorphoNumber) : MorphoFeaturesExtractor {

  /**
   * Return a list of features.
   */
  override fun get(): List<String> = listOf(
    "p:%s".format(this.morphology.pos),
    "p:%s n:%s g:%s".format(
      this.morphology.pos,
      this.morphology.number,
      this.morphology.gender)
  )
}
