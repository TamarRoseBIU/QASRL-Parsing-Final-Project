name := "qasrl-slot-filling"

version := "0.1"

scalaVersion := "2.12.18"

resolvers ++= Seq(
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  "Sonatype OSS Releases" at "https://oss.sonatype.org/content/repositories/releases"
)

libraryDependencies ++= Seq(
  "org.julianmichael" %% "qasrl" % "0.1.0",
  "com.github.tototoshi" %% "scala-csv" % "1.3.10",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
  "ch.qos.logback" % "logback-classic" % "1.4.11"
)