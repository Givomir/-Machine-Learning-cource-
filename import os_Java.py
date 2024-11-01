import os
import jpype

# Set JAVA_HOME to the path of the Java package
java_path = jpype.getDefaultJVMPath()
os.environ["JAVA_HOME"] = java_path
os.environ["PATH"] += os.pathsep + java_pathcd