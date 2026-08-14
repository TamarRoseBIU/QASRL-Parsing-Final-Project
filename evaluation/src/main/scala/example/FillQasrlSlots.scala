package example

import java.nio.file.Paths
import com.github.tototoshi.csv.{CSVReader, CSVWriter}
import com.typesafe.scalalogging.StrictLogging
import nlpdata.datasets.wiktionary.{InflectedForms, Inflections, WiktionaryFileSystemService}
import nlpdata.util.LowerCaseStrings._
import qasrl.labeling.SlotBasedLabel
import qasrl.{QuestionProcessor, TemplateStateMachine, Frame, ArgumentSlot}
import qasrl.util.DependentMap
import cats.implicits._
import cats.Id

case class QAPrediction(
  qasrlId: String, 
  verbIdx: Int, 
  verb: String, 
  question: String, 
  answerRange: String,
  answer: String
)

object FillQasrlSlots extends App with StrictLogging {

  // ============================================================
  // COMMAND LINE ARGUMENTS
  // ============================================================
  if (args.length != 3) {
    println("=" * 70)
    println("ERROR: Incorrect number of arguments")
    println("=" * 70)
    println("Usage:")
    println("  sbt \"runMain example.FillQasrlSlots <predictions.csv> <sentences.csv> <output.csv>\"")
    println()
    println("Arguments:")
    println("  1. predictions.csv - Python inference output")
    println("  2. sentences.csv   - Original input CSV")
    println("  3. output.csv      - Output path for slot-filled predictions")
    println("=" * 70)
    System.exit(1)
  }

  val pythonOutputPath = Paths.get(args(0))
  val sentencesPath = Paths.get(args(1))
  val finalOutputPath = Paths.get(args(2))

  println("=" * 70)
  println("QA-SRL SLOT FILLING")
  println("=" * 70)
  println(s"Reading predictions from: $pythonOutputPath")
  println(s"Reading sentences from:   $sentencesPath")
  println(s"Writing output to:        $finalOutputPath")
  println("=" * 70)

  val predictionsReader = CSVReader.open(pythonOutputPath.toString)
  val sentReader = CSVReader.open(sentencesPath.toString)
  val outputWriter = CSVWriter.open(finalOutputPath.toString)

  // ============================================================
  // LOAD WIKTIONARY FOR VERB INFLECTIONS
  // ============================================================
  val wiktionaryPath = Paths.get("datasets/wiktionary")
  if (!wiktionaryPath.toFile.exists()) {
    logger.error(s"Wiktionary dataset not found at: $wiktionaryPath")
    logger.error("Please ensure Wiktionary is installed at datasets/wiktionary/")
    System.exit(1)
  }
  val wiktionary = new WiktionaryFileSystemService(wiktionaryPath)

  // ============================================================
  // LOAD SENTENCES AND TOKENS
  // ============================================================
  val sentMap: Map[String, Vector[String]] = sentReader.allWithHeaders().map { rec =>
    val qasrlId = rec("qasrl_id")
    val tokensStr = rec("tokens")
    
    val tokens = if (tokensStr.startsWith("[")) {
      tokensStr
        .stripPrefix("[")
        .stripSuffix("]")
        .split(",")
        .map(_.trim.stripPrefix("'").stripSuffix("'").stripPrefix("\"").stripSuffix("\""))
        .toVector
    } else {
      tokensStr.split(" ").toVector
    }
    
    qasrlId -> tokens
  }.toMap

  println(s"Loaded ${sentMap.size} sentences")

  // ============================================================
  // LOAD PREDICTIONS FROM PYTHON
  // ============================================================
  val predictions: Vector[QAPrediction] = (for (rec <- predictionsReader.allWithHeaders())
    yield QAPrediction(
      rec("qasrl_id"),
      rec("verb_idx").toInt,
      rec("verb"),
      rec("question"),
      rec("answer_range"),
      rec("answer")
    )).toVector

  println(s"Loaded ${predictions.size} predictions")

  // ============================================================
  // WRITE OUTPUT CSV HEADER
  // ============================================================
  outputWriter.writeRow(List(
    "qasrl_id", "verb_idx", "verb", "question", 
    "answer_range", "answer",
    "wh", "subj", "obj", "obj2", "aux", "prep", "verb_prefix",
    "is_passive", "is_negated"
  ))

  var processedCount = 0
  var successCount = 0
  var failedSentences = 0

  // ============================================================
  // PROCESS PREDICTIONS - GROUPED BY SENTENCE AND VERB
  // ============================================================
  for ((qasrlId, sentPredictions) <- predictions.groupBy(_.qasrlId)) {
    val tokensOpt = sentMap.get(qasrlId)
    
    tokensOpt match {
      case None =>
        logger.warn(s"Missing tokens for $qasrlId - skipping ${sentPredictions.size} predictions")
        failedSentences += 1
        for (pred <- sentPredictions) {
          outputWriter.writeRow(List(
            pred.qasrlId, pred.verbIdx.toString, pred.verb, pred.question,
            pred.answerRange, pred.answer,
            "_", "_", "_", "_", "_", "_", "_", "False", "False"
          ))
          processedCount += 1
        }
        
      case Some(tokens) =>
        val inflections = wiktionary.getInflectionsForTokens(tokens.iterator)
        
        for ((verbIdx, predRecords) <- sentPredictions.groupBy(_.verbIdx)) {
          if (verbIdx >= tokens.length) {
            logger.warn(s"Verb index $verbIdx out of bounds for sentence with ${tokens.length} tokens at $qasrlId - skipping ${predRecords.size} predictions")
            for (pred <- predRecords) {
              outputWriter.writeRow(List(
                pred.qasrlId, pred.verbIdx.toString, pred.verb, pred.question,
                pred.answerRange, pred.answer,
                "_", "_", "_", "_", "_", "_", "_", "False", "False"
              ))
              processedCount += 1
            }
          } else {
            val verb = predRecords.head.verb
          
          
          // Get verb inflections - create simple inflections if not found
          val verbInflectedFormsOpt = inflections.getInflectedForms(verb.lowerCase)
          val verbInflectedForms = verbInflectedFormsOpt.getOrElse {
            logger.warn(s"No inflections for verb '$verb' at $qasrlId:$verbIdx - using basic form")
            // Create a basic InflectedForms with just the verb stem
            InflectedForms(
              stem = verb.lowerCase,
              present = verb.lowerCase,
              presentParticiple = verb.lowerCase,
              past = verb.lowerCase,
              pastParticiple = verb.lowerCase
            )
          }
          
          for (pred <- predRecords) {
            processedCount += 1
            val question = pred.question
            
            // ============================================================
            // PARSE QUESTION TO EXTRACT SLOTS
            // ============================================================
            val qTokens = question.init.split(" ").toVector.map(_.lowerCase)
            val qPreps = qTokens.filter(TemplateStateMachine.allPrepositions.contains).toSet
            val qPrepBigrams = qTokens.sliding(2)
              .filter(_.forall(TemplateStateMachine.allPrepositions.contains))
              .map(_.mkString(" ").lowerCase)
              .toSet

            val stateMachine = new TemplateStateMachine(
              tokens,
              verbInflectedForms,
              Some(qPreps ++ qPrepBigrams)
            )
            val template = new QuestionProcessor(stateMachine)

            // Parse question through the template state machine
            val goodStatesOpt = template.processStringFully(question).toOption
            
            // Extract slots - getSlotsForQuestion returns Option[SlotBasedLabel]
            val slotOpt = SlotBasedLabel.getSlotsForQuestion(
              tokens, 
              verbInflectedForms, 
              List(question)
            ).headOption.flatten
            
            // ============================================================
            // EXTRACT FRAME INFO
            // ============================================================
            val frameOpt = goodStatesOpt.flatMap { goodStates =>
              goodStates.toList.collectFirst {
                case QuestionProcessor.CompleteState(_, frame, _) => frame
              }
            }

            // ============================================================
            // WRITE ROW TO OUTPUT CSV
            // ============================================================
            (slotOpt, frameOpt) match {
              case (Some(slot), Some(frame)) =>
                // Successfully parsed - write row with all slots from actual frame
                outputWriter.writeRow(List(
                  pred.qasrlId,
                  pred.verbIdx.toString,
                  pred.verb,
                  pred.question,
                  pred.answerRange,
                  pred.answer,
                  slot.wh.toString,
                  slot.subj.getOrElse("_"),
                  slot.obj.getOrElse("_"),
                  slot.obj2.getOrElse("_"),
                  slot.aux.getOrElse("_"),
                  slot.prep.getOrElse("_"),
                  if (slot.verbPrefix.isEmpty) "_" else slot.verbPrefix,
                  if (frame.isPassive) "True" else "False",
                  if (frame.isNegated) "True" else "False"
                ))
                successCount += 1
              
              case (Some(slot), None) =>
                // Have slot but no frame - use defaults for frame fields
                outputWriter.writeRow(List(
                  pred.qasrlId,
                  pred.verbIdx.toString,
                  pred.verb,
                  pred.question,
                  pred.answerRange,
                  pred.answer,
                  slot.wh.toString,
                  slot.subj.getOrElse("_"),
                  slot.obj.getOrElse("_"),
                  slot.obj2.getOrElse("_"),
                  slot.aux.getOrElse("_"),
                  slot.prep.getOrElse("_"),
                  if (slot.verbPrefix.isEmpty) "_" else slot.verbPrefix,
                  "False",
                  "False"
                ))
                successCount += 1
                
              case _ =>
                // Failed to parse - write row with empty slots
                logger.warn(s"Failed to parse: '$question' for $qasrlId:$verbIdx")
                outputWriter.writeRow(List(
                  pred.qasrlId,
                  pred.verbIdx.toString,
                  pred.verb,
                  pred.question,
                  pred.answerRange,
                  pred.answer,
                  "_", "_", "_", "_", "_", "_", "_",
                  "False", "False"
                ))
            }
          }
        }
      }
    }
  }

  // ============================================================
  // CLEANUP AND STATISTICS
  // ============================================================
  predictionsReader.close()
  sentReader.close()
  outputWriter.close()

  println()
  println("=" * 70)
  println("SLOT FILLING COMPLETE")
  println("=" * 70)
  println(f"Total predictions:    $processedCount%,d")
  println(f"Successfully parsed:  $successCount%,d (${successCount * 100.0 / processedCount}%.1f%%)")
  println(f"Failed to parse:      ${processedCount - successCount}%,d")
  println(f"Missing sentences:    $failedSentences")
  println("=" * 70)
  println(s"Output saved to: $finalOutputPath")
  println("=" * 70)
}