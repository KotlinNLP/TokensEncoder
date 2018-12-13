/* Copyright 2018-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * -----------------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings.keyextractor

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken

/**
 * An [EmbeddingKeyExtractor] by normalized form.
 */
class NormWordKeyExtractor : EmbeddingKeyExtractor<FormToken, Sentence<FormToken>> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * @param sentence a generic sentence
   * @param tokenId the id of the token from which to extract the key
   *
   * @return the form of the token
   */
  override fun getKey(sentence: Sentence<FormToken>, tokenId: Int): String = sentence.tokens[tokenId].normalizedForm
}
