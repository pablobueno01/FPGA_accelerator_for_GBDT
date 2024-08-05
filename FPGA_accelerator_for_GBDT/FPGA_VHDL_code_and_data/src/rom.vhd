-------------------------------------------------------------------------------
-- Synchronous ROM with generic memory and data sizes
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity rom is
    generic(ADDRESS_BITS: positive;
            DATA_LENGTH:  positive);
    port(-- Control signals
         Clk: in std_logic;
         Re:  in std_logic;
         
         -- Input signals
         Addr: in std_logic_vector (ADDRESS_BITS - 1 downto 0);
         
         -- Output
         Dout: out std_logic_vector (DATA_LENGTH - 1 downto 0));
end rom;

architecture Behavioral of rom is

type MemoryBank is array(0 to 2**ADDRESS_BITS - 1)
                   of std_logic_vector(DATA_LENGTH - 1 downto 0);
constant bank: MemoryBank := (
    -- Initialize the ROM data here
    0 => x"00010003",
    1 => x"00010003",
    2 => x"00010003",
    others => (others => '0')
);

begin
process (Clk)
begin
    if rising_edge(Clk) then
        if (Re = '1') then
            -- Read from Addr
            Dout <= bank(to_integer(unsigned(Addr)));
        else
            Dout <= (others => '0');
        end if;
    end if;
end process;
end Behavioral;