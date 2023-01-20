#!/usr/bin/perl

($threshold, $scaletrue, $scalefalse) = @ARGV;

while($line = <STDIN>)
{
	chomp($line);
    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;
    @chunk = split(/\s+/, $line);
    $name = $chunk[0];
    $len = @chunk;
    print ("$name [");
    for ($i=2;$i<$len-1;$i++) {
        if ($chunk[$i] < $threshold) {
            if ($scalefalse eq "") {
                $scalefalse = 0;
            }
            print (" $scalefalse");
        } else {
            if ($scaletrue eq "") {
                $scaletrue = $chunk[$i];
            }
            print (" $scaletrue");
        }
    }
    print (" ]\n");
}

