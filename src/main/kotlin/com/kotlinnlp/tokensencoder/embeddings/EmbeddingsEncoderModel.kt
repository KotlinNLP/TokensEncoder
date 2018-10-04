/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMapByDictionary
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.UpdateMethod
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.EmbeddingKeyExtractor

/**
 * The model of the [EmbeddingsEncoder].
 *
 * @param embeddingsMap the embeddings map
 * @param dropoutCoefficient the dropout coefficient
 * @param embeddingKeyExtractor an embeddings key extractor
 */
class EmbeddingsEncoderModel<TokenType: Token, SentenceType: Sentence<TokenType>>(
  val embeddingsMap: EmbeddingsMapByDictionary,
  val dropoutCoefficient: Double = 0.0,
  val embeddingKeyExtractor: EmbeddingKeyExtractor<TokenType, SentenceType>
) : TokensEncoderModel<TokenType, SentenceType> {

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
  override val tokenEncodingSize: Int = this.embeddingsMap.size

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d (dropout %.2f)
  """.trimIndent().format(
    this.tokenEncodingSize,
    this.dropoutCoefficient
  )

  /**
   * @param useDropout whether to apply the dropout
   * @param id an identification number useful to track a specific encoder
   *
   * @return a new tokens encoder that uses this model
   */
  override fun buildEncoder(useDropout: Boolean, id: Int) = EmbeddingsEncoder(
    model = this,
    useDropout = useDropout,
    id = id
  )

  /**
   * @param updateMethod the update method helper (Learning Rate, ADAM, AdaGrad, ...)
   *
   * @return a new optimizer for this model
   */
  override fun buildOptimizer(updateMethod: UpdateMethod<*>) = EmbeddingsEncoderOptimizer(
    model = this,
    updateMethod = updateMethod
  )
}
