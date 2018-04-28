/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings.dictionary

import com.kotlinnlp.simplednn.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.tokensencoder.embeddings.EmbeddingsEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.TokenEmbeddingKey

/**
 * The model of the [EmbeddingsEncoderByDictionary].
 *
 * @param embeddingsMap the embeddings map
 * @property tokenEmbeddingKey the extractor of a token property to be used as embeddings key
 * @property dropoutCoefficient the embeddings dropout coefficient
 */
class EmbeddingsEncoderByDictionaryModel(
  override val embeddingsMap: EmbeddingsMapByDictionary,
  val tokenEmbeddingKey: TokenEmbeddingKey,
  val dropoutCoefficient: Double = 0.0
) : EmbeddingsEncoderModel(wordEmbeddingSize = embeddingsMap.size) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d (dropout %.2f)
  """.trimIndent().format(
    this.tokenEncodingSize,
    this.dropoutCoefficient
  )
}
