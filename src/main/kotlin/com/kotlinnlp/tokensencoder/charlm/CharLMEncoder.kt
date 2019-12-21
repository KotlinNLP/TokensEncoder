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
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
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
) : TokensEncoder<FormToken, Sentence<FormToken>>() {

  /**
   * The processing sentence.
   *
   * @property tokens the list of tokens
   */
  private class ProcessingSentence(override val tokens: List<FormToken>) : Sentence<FormToken> {

    /**
     * @property direct the index of the end of a token in the left-to-right sequence
     * @property reverse the index of the end of a token in the right-to-left sequence
     */
    data class TokenEnd(val direct: Int, val reverse: Int)

    /**
     * Secondary constructor.
     *
     * @param sentence a sentence of [FormToken]
     */
    constructor(sentence: Sentence<FormToken>): this(sentence.tokens)

    /**
     * String composed by the sentence tokens separated by space.
     */
    val spaceSeparatedForms: String get() = this.tokens.joinToString(" ") { it.form }

    /**
     * The indexes of the tokens ends within the sequence, seen left-to-right (direct) and right-to-left (reverse).
     */
    val tokensEnds: List<TokenEnd>

    /**
     * Initialize the tokens ends.
     */
    init {

      var tokenStart = 0

      this.tokensEnds = this.tokens.map {
        TokenEnd(
          direct = tokenStart + it.form.lastIndex,
          reverse = this.spaceSeparatedForms.lastIndex - tokenStart
        ).also { e ->
          tokenStart = e.direct + 2 // space included
        }
      }
    }
  }

  /**
   * Don't use the dropout.
   */
  override val useDropout: Boolean = false

  /**
   * The hidden recurrent processor that auto-encodes the sequence from left to right.
   */
  private val directProcessor = RecurrentNeuralProcessor<DenseNDArray>(
    model = this.model.dirCharLM.recurrentNetwork,
    useDropout = false,
    propagateToInput = false)

  /**
   * The hidden recurrent processor that auto-encodes the sequence from right to left.
   */
  private val reverseProcessor = RecurrentNeuralProcessor<DenseNDArray>(
    model = this.model.revCharLM.recurrentNetwork,
    useDropout = false,
    propagateToInput = false)

  /**
   * The processor that merges the encoded vectors.
   */
  private val outputMergeProcessors = BatchFeedforwardProcessor<DenseNDArray>(
    model = this.model.outputMergeNetwork,
    useDropout = false, // TODO: why don't use the dropout here?
    propagateToInput = false)

  /**
   * The current [forward]ed input.
   **/
  private lateinit var curSentence: ProcessingSentence

  /**
   * The Forward.
   *
   * @param input the input
   *
   * @return the result of the forward
   */
  override fun forward(input: Sentence<FormToken>): List<DenseNDArray> {

    this.curSentence = ProcessingSentence(input)

    val inputDirect: List<DenseNDArray>
    val inputReverse: List<DenseNDArray>

    this.curSentence.spaceSeparatedForms.let { s ->
      inputDirect = s.map { this.model.dirCharLM.charsEmbeddings[it].values }
      inputReverse = s.map { this.model.revCharLM.charsEmbeddings[it].values }.reversed()
    }

    val hiddenDirect: List<DenseNDArray> = this.directProcessor.forward(inputDirect)
    val hiddenReverse: List<DenseNDArray> = this.reverseProcessor.forward(inputReverse)

    return this.outputMergeProcessors.forward(ArrayList(this.curSentence.tokensEnds.map {
      listOf(hiddenDirect[it.direct], hiddenReverse[it.reverse])
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
  override fun getParamsErrors(copy: Boolean): ParamsErrorsList =
    this.outputMergeProcessors.getParamsErrors(copy = copy)
}
