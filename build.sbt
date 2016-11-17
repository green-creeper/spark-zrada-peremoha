name := """spark-zrada"""

version := "1.0"

scalaVersion := "2.11.8"
spName := "organization/zp"
sparkVersion := "2.0.1"
// Change this to another test framework if you prefer
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

// Uncomment to use Akka
//libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.3.11"

libraryDependencies += "org.telegram" % "telegrambots" % "2.4.0"
sparkComponents ++= Seq("mllib", "sql")
