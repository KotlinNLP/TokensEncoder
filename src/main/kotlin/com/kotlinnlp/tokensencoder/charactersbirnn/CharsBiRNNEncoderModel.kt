/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charactersbirnn

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.core.embeddings.EmbeddingsMap
import com.kotlinnlp.tokensencoder.TokensEncoderModel

/**
 * The model of the [CharsBiRNNEncoder].
 *
 * @param words the list of words from which to extract the list of possible characters
 * @param charEmbeddingSize the size of the character embeddings
 * @param hiddenSize the hidden size of the BiRNN
 * @param connectionType the layer connection type of the BiRNN
 * @param hiddenActivation the activation function of the BiRNN
 */
class CharsBiRNNEncoderModel(
  private val words: List<String>,
  charEmbeddingSize: Int = 25,
  hiddenSize: Int = 25,
  connectionType: LayerType.Connection = LayerType.Connection.LSTM,
  hiddenActivation: ActivationFunction? = Tanh(),
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : TokensEncoderModel<FormToken, Sentence<FormToken>> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The chars embeddings.
   */
  val charsEmbeddings = EmbeddingsMap<Char>(size = charEmbeddingSize)

  /**
   * The BiRNN of the ContextEncoder.
   */
  val biRNN = BiRNN(
    inputType = LayerType.Input.Dense,
    inputSize = charEmbeddingSize,
    recurrentConnectionType = connectionType,
    hiddenActivation = hiddenActivation,
    hiddenSize = hiddenSize,
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The size of the token encoding vectors.
   */
  override val tokenEncodingSize: Int = this.biRNN.outputSize

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
    encoding size %d [chars emb. size %d, hidden size %d, activation %s - %s]
  """.trimIndent().format(
    this.tokenEncodingSize,
    this.charsEmbeddings.size,
    this.biRNN.hiddenSize,
    if (this.biRNN.hiddenActivation != null)
      this.biRNN.hiddenActivation!!::class.simpleName
    else
      "None",
    this.biRNN.recurrentConnectionType
  )
}
