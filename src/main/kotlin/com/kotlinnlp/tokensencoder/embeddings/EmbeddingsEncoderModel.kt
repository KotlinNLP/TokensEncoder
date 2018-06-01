/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import java.io.Serializable

/**
 * The model of the [EmbeddingsEncoder].
 *
 * @param wordEmbeddingSize the size of each word embedding vector
 */
abstract class EmbeddingsEncoderModel(
  private val wordEmbeddingSize: Int
) : TokensEncoderModel, Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int = this.wordEmbeddingSize

  /**
   * The word embeddings.
   */
  abstract val embeddingsMap: EmbeddingsMap<*>

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d
  """.trimIndent().format(
    this.tokenEncodingSize
  )
}
