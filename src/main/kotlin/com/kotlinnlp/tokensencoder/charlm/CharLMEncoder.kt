/* Copyright 2017-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.tokensencoder.charlm

import com.kotlinnlp.linguisticdescription.sentence.Sentence
import com.kotlinnlp.linguisticdescription.sentence.token.FormToken
import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward.BatchFeedforwardProcessor
import com.kotlinnlp.simplednn.core.neuralprocessor.recurrent.RecurrentNeuralProcessor
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.tokensencoder.TokensEncoder

/**
 * The [TokensEncoder] that encodes a token using a the hidden states of two characters language models.
 *
 * @property model the model of this tokens encoder
 * @property id an identification number useful to track a specific processor*
 */
class CharLMEncoder(
  override val model: CharLMEncoderModel,
  override val id: Int = 0
) : TokensEncoder<FormToken, Sentence<FormToken>>(model) {

  /**
   * Don't use the dropout.
   */
  override val useDropout: Boolean = false

  /**
   * The recurrent processor.
   */
  private val leftToRightProcessor = RecurrentNeuralProcessor<DenseNDArray>(
    neuralNetwork = this.model.charLM.recurrentNetwork,
    useDropout = false,
    propagateToInput = false)

  /**
   * The recurrent processor.
   */
  private val rightToLeftProcessor = RecurrentNeuralProcessor<DenseNDArray>(
    neuralNetwork = this.model.revCharLM.recurrentNetwork,
    useDropout = false,
    propagateToInput = false)

  /**
   * The processor that merges the encoded vectors.
   */
  private val outputMergeProcessors = BatchFeedforwardProcessor<DenseNDArray>(
    neuralNetwork = this.model.outputMergeNetwork,
    useDropout = false, // TODO: why don't use the dropout here?
    propagateToInput = false)

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: Sentence<FormToken>): List<DenseNDArray> {

    val s = input.tokens.joinToString(" ") { it.form }

    val inputL2R: List<DenseNDArray> = s.map { this.model.charLM.charsEmbeddings[it].array.values }
    val inputR2L: List<DenseNDArray> = s.map { this.model.revCharLM.charsEmbeddings[it].array.values }.reversed()

    val hiddenL2R: List<DenseNDArray> = this.leftToRightProcessor.forward(inputL2R)
    val hiddenR2L: List<DenseNDArray> = this.rightToLeftProcessor.forward(inputR2L)

    var tokenStart = 0

    return this.outputMergeProcessors.forward(ArrayList(input.tokens.map {

      val tokenEnd = tokenStart + it.form.lastIndex
      val reverseEnd = s.lastIndex - tokenStart

      tokenStart = tokenEnd + 2 // + 1 to include the spaces

      listOf(hiddenL2R[tokenEnd], hiddenR2L[reverseEnd])
    }))
  }

  /**
   * The Backward.
   *
   * @param outputErrors the output errors
   */
  override fun backward(outputErrors: List<DenseNDArray>) = this.outputMergeProcessors.backward(outputErrors)

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  override fun getInputErrors(copy: Boolean) = NeuralProcessor.NoInputErrors

  /**
   * Return the params errors of the last backward.
   *
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  override fun getParamsErrors(copy: Boolean) =
    CharLMEncoderParams(this.outputMergeProcessors.getParamsErrors(copy = copy))
}
