#pragma once

// Keys are prefixed by the producing render pass name in the graph dictionary.
namespace ReconstructionResult
{
inline constexpr char kReady[] = ".resultReady";
inline constexpr char kScene[] = ".resultScene";
inline constexpr char kCamera[] = ".resultCamera";
inline constexpr char kFlipValid[] = ".outputValid";
}
