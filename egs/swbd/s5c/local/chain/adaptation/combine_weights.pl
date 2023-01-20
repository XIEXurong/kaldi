#!/usr/bin/perl

($weight2,$scale1,$scale2,$avg) = @ARGV;

open(IN, $weight2);

while($line = <STDIN>)
{
	chomp($line);
    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;
    @chunk = split(/\s+/, $line);
    $name = $chunk[0];
    $len = @chunk;
    
    $line2 = <IN>;
    chomp($line2);
    $line2 =~ s/^\s+//g;
    $line2 =~ s/\s+$//g;
    @chunk2 = split(/\s+/, $line2);
    $name2 = $chunk2[0];
    $len2 = @chunk2;
    
    if ($name ne $name2) {
        print ("Name $name is not equal to $name2");
        exit;
    }
    if ($len != $len2) {
        print ("length $len is not equal to $len2");
        exit;
    }
    
    print ("$name [");
    for ($i=2;$i<$len-1;$i++) {
        if ($avg eq "true") {
            $tmp = $scale1 * $chunk[$i] + $scale2 * $chunk2[$i];
        } else {
            $tmp = ($chunk[$i]**$scale1) * ($chunk2[$i]**$scale2);
        }
        print " $tmp";
    }
    print (" ]\n");
}

close IN;
