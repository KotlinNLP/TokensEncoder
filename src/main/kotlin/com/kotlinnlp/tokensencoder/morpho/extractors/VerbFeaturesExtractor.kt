/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.morpho.extractors

import com.kotlinnlp.linguisticdescription.morphology.morphologies.relations.Verb

/**
 * Extract the features from the given [morphology].
 *
 * @param morphology the morphology
 */
class VerbFeaturesExtractor(private val morphology: Verb) : MorphoFeaturesExtractor {

  /**
   * Return a list of features.
   */
  override fun get(): List<String> = listOf(
    "t:%s".format("Verb"),
    "p:%s".format(this.morphology.type),
    "p:%s m:%s".format(this.morphology.type, this.morphology.mood),
    "p:%s m:%s t:%s".format(this.morphology.type, this.morphology.mood, this.morphology.tense),
    "p:%s m:%s n:%s p:%s g:%s".format(
      morphology.type,
      morphology.mood,
      morphology.person,
      morphology.number,
      morphology.gender)
  )
}

