/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersattention

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.attention.han.HAN
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.TokensEncoderModel
import java.io.Serializable

/**
 * The model of the [CharsAttentionEncoder].
 *
 * @param words the list of words from which to extract the list of possible characters
 * @param charEmbeddingSize the size of the character embeddings
 * @param wordEmbeddingSize the size of the token encoding vectors
 * @param hanAttentionSize the size of the attention core.arrays of the HAN attention core.layers
 * @param hanConnectionType the layer connection type of the HAN BiRNN
 * @param hanHiddenActivation the activation function of the HAN BiRNN
 */
class CharsAttentionEncoderModel(
  private val words: List<String>,
  charEmbeddingSize: Int = 25,
  wordEmbeddingSize: Int = 50,
  hanAttentionSize: Int = 25,
  hanConnectionType: LayerType.Connection = LayerType.Connection.LSTM,
  hanHiddenActivation: ActivationFunction? = Tanh()
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
  override val tokenEncodingSize: Int = wordEmbeddingSize

  /**
   * The chars embeddings.
   */
  val charsEmbeddings = EmbeddingsMap<Char>(size = charEmbeddingSize)

  /**
   * A [HAN] model.
   */
  val charactersNetwork = HAN(
    hierarchySize = 1,
    inputSize = charEmbeddingSize,
    inputType = LayerType.Input.Dense,
    biRNNsActivation = hanHiddenActivation,
    biRNNsConnectionType = hanConnectionType,
    attentionSize = hanAttentionSize,
    outputSize = wordEmbeddingSize,
    outputActivation = null,
    gainFactors = arrayOf(2.0))

  /**
   * Initialize chars embeddings.
   */
  init { this.words.forEach { this.charsEmbeddings.includeChars(it) } }

  /**
   * Include in the map the characters of the given [word].
   *
   * @param word a string
   */
  private fun EmbeddingsMap<Char>.includeChars(word: String) =
    word.filterNot { this.contains(it) }.forEach { this.set(it) }

  /**
   * @return the string representation of this model
   */
  override fun toString(): String = """
    encoding size %d [chars emb. size %d, HAN: attention %d - activation %s - %s]
  """.trimIndent().format(
    this.tokenEncodingSize,
    this.charsEmbeddings.size,
    this.charactersNetwork.attentionSize,
    if (this.charactersNetwork.biRNNsActivation != null)
      this.charactersNetwork.biRNNsActivation!!::class.simpleName
    else
      "None",
    this.charactersNetwork.biRNNsConnectionType
  )
}
