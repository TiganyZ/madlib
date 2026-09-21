# Install/unInstall package files in LAMMPS
# mode = 0/1/2 for uninstall/install/update

mode=$1

for file in *.cpp *.h; do
  if (test $mode = 0) then
    rm -f ../$file
  elif (test ! -e ../$file) then
    ln -s ../MADLIB/$file ..
  fi
done
