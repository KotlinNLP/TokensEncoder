/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.embeddings

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.Token
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import com.kotlinnlp.tokensencoder.embeddings.keyextractor.EmbeddingKeyExtractor

/**
 * The model of the [EmbeddingsEncoder].
 *
 * @property embeddingsMap an embeddings map
 * @property frequencyDictionary a map of words to the number of their occurrences
 * @property dropout the dropout [0.0 .. 1.0]. When the [frequencyDictionary] is not null, the dropout is considered
 *                   as a coefficient to calculate the probability of the final dropout probability
 * @param embeddingKeyExtractor list of embeddings key extractor
 * @param fallbackEmbeddingKeyExtractors list of embeddings key extractors sorted by priority in descending order,
 *                                       used in case the principal extractor does not generate a valid key
 */
sealed class EmbeddingsEncoderModel<TokenType: Token, SentenceType: Sentence<TokenType>>(
  internal val frequencyDictionary: Map<String, Int>? = null,
  internal val dropout: Double = 0.0,
  embeddingKeyExtractor: EmbeddingKeyExtractor<TokenType, SentenceType>,
  fallbackEmbeddingKeyExtractors: List<EmbeddingKeyExtractor<TokenType, SentenceType>> = emptyList()
) : TokensEncoderModel<TokenType, SentenceType> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  init { require(dropout in 0.0 .. 1.0) }

  /**
   * An embeddings map.
   */
  abstract val embeddingsMap: EmbeddingsMap<String>

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int by lazy { this.embeddingsMap.size }

  /**
   * The list of embeddings key extractors, sorted by priority in descending order.
   */
  internal val keyExtractors = listOf(embeddingKeyExtractor) + fallbackEmbeddingKeyExtractors

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d (dropout %.2f)
  """.trimIndent().format(
    this.tokenEncodingSize,
    this.dropout
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
   * The base model of the [EmbeddingsEncoder].
   *
   * @property embeddingsMap an embeddings map
   * @property frequencyDictionary map of elements with their relative absolute frequency in a corpus
   * @property dropout the dropout [0.0 .. 1.0]. When the [frequencyDictionary] is not null, the dropout is considered as
   *                   a coefficient to calculate the probability of the final dropout probability
   * @param embeddingKeyExtractor list of embeddings key extractor
   * @param fallbackEmbeddingKeyExtractors list of embeddings key extractors sorted by priority in descending order,
   *                                       used in case the principal extractor does not generate a valid key
   */
  class Base<TokenType: Token, SentenceType: Sentence<TokenType>>(
    override val embeddingsMap: EmbeddingsMap<String>,
    frequencyDictionary: Map<String, Int>? = null,
    dropout: Double = 0.0,
    embeddingKeyExtractor: EmbeddingKeyExtractor<TokenType, SentenceType>,
    fallbackEmbeddingKeyExtractors: List<EmbeddingKeyExtractor<TokenType, SentenceType>> = emptyList()
  ) : EmbeddingsEncoderModel<TokenType, SentenceType>(
    frequencyDictionary = frequencyDictionary,
    dropout = dropout,
    embeddingKeyExtractor = embeddingKeyExtractor,
    fallbackEmbeddingKeyExtractors = fallbackEmbeddingKeyExtractors
  ) {

    companion object {

      /**
       * Private val used to serialize the class (needed by Serializable).
       */
      @Suppress("unused")
      private const val serialVersionUID: Long = 1L
    }
  }

  /**
   * The model of the [EmbeddingsEncoder] with a transient embeddings map, which is not included in the serialization.
   *
   * @param embeddingsMap an embeddings map
   * @property dropout the dropout [0.0 .. 1.0]
   * @param embeddingKeyExtractor list of embeddings key extractor
   * @param fallbackEmbeddingKeyExtractors list of embeddings key extractors sorted by priority in descending order,
   *                                       used in case the principal extractor does not generate a valid key
   */
  class Transient<TokenType: Token, SentenceType: Sentence<TokenType>>(
    embeddingsMap: EmbeddingsMap<String>,
    dropout: Double = 0.0,
    embeddingKeyExtractor: EmbeddingKeyExtractor<TokenType, SentenceType>,
    fallbackEmbeddingKeyExtractors: List<EmbeddingKeyExtractor<TokenType, SentenceType>> = emptyList()
  ) : EmbeddingsEncoderModel<TokenType, SentenceType>(
    frequencyDictionary = null,
    dropout = dropout,
    embeddingKeyExtractor = embeddingKeyExtractor,
    fallbackEmbeddingKeyExtractors = fallbackEmbeddingKeyExtractors
  ) {

    companion object {

      /**
       * Private val used to serialize the class (needed by Serializable).
       */
      @Suppress("unused")
      private const val serialVersionUID: Long = 1L
    }

    /**
     * An embeddings map.
     */
    override val embeddingsMap: EmbeddingsMap<String> get() = checkNotNull(this.embeddingsMapTransient) {
      "The embeddings map must be set to use the Transient Embeddings Encoder Model."
    }

    /**
     * The transient embeddings map of this model, which will not be serialized.
     */
    @kotlin.jvm.Transient private var embeddingsMapTransient: EmbeddingsMap<String>? = embeddingsMap

    /**
     * Set the embeddings map of this model.
     *
     * @param embeddingsMap an embeddings map
     */
    fun setEmbeddingsMap(embeddingsMap: EmbeddingsMap<String>) {
      this.embeddingsMapTransient = embeddingsMap
    }
  }
}